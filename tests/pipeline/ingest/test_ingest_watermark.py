"""Watermark and locking semantics.

These are the rules that decide whether an unattended ingester recovers from an
outage or quietly stops keeping up.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.pool import NullPool

from hontology.db.session import session_scope
from hontology.pipeline.ingest.feed import catchup, gdelt, slices, store, watermark
from hontology.pipeline.ingest.feed.scheduler import try_lock, unlock

pytestmark = pytest.mark.requires_db

FEED = "test_feed"


def set_slice_status(session, key: str, status: str) -> None:
    row = store.get_or_create_slice(session, key, FEED)
    row.status = status
    session.flush()


class TestWatermarkAdvance:
    def test_first_advance_accepts_any_slice(self):
        with session_scope() as session:
            assert watermark.advance_watermark(session, "20260826120000", FEED) is True
            assert watermark.get_watermark(session, FEED).last_slice_key == "20260826120000"

    def test_advances_to_the_immediate_successor(self):
        with session_scope() as session:
            watermark.advance_watermark(session, "20260826120000", FEED)
            assert watermark.advance_watermark(session, "20260826121500", FEED) is True
            assert watermark.get_watermark(session, FEED).last_slice_key == "20260826121500"

    def test_refuses_to_skip_a_slice(self):
        """A later success must not mark an earlier unfinished slice as done."""
        with session_scope() as session:
            watermark.advance_watermark(session, "20260826120000", FEED)
            # 12:15 was never processed; jumping to 12:30 would silently lose it.
            assert watermark.advance_watermark(session, "20260826123000", FEED) is False
            assert watermark.get_watermark(session, FEED).last_slice_key == "20260826120000"

    def test_refuses_to_move_backwards(self):
        with session_scope() as session:
            watermark.advance_watermark(session, "20260826123000", FEED)
            assert watermark.advance_watermark(session, "20260826120000", FEED) is False
            assert watermark.get_watermark(session, FEED).last_slice_key == "20260826123000"

    def test_re_advancing_to_the_same_slice_is_refused(self):
        with session_scope() as session:
            watermark.advance_watermark(session, "20260826120000", FEED)
            assert watermark.advance_watermark(session, "20260826120000", FEED) is False


class TestSliceStatus:
    def test_a_terminal_slice_is_skipped(self):
        with session_scope() as session:
            set_slice_status(session, "20260826120000", "ok")
        with session_scope() as session:
            result = slices.ingest_slice(session, "20260826120000", feed=FEED)
            assert result["skipped"] is True
            assert result["status"] == "ok"

    @pytest.mark.parametrize("status", ["ok", "empty", "missing"])
    def test_terminal_statuses_let_the_watermark_move(self, status):
        """A permanently skipped slice must not stall the feed forever."""
        assert status in slices.TERMINAL

    def test_failed_is_not_terminal(self):
        """Transport failures must be retried, not stepped over."""
        assert "failed" not in slices.TERMINAL
        assert "pending" not in slices.TERMINAL

    def test_a_failed_slice_is_retried_rather_than_skipped(self):
        with session_scope() as session:
            set_slice_status(session, "20260826120000", "failed")
        with session_scope() as session:
            # Not short-circuited: it will attempt the fetch again.
            row = store.get_or_create_slice(session, "20260826120000", FEED)
            assert row.status not in slices.TERMINAL


class TestBackfill:
    def test_backfill_does_not_move_the_watermark(self):
        """Backfilling history must not convince the scheduler it is current."""
        with session_scope() as session:
            watermark.advance_watermark(session, "20260826120000", FEED)

        with session_scope() as session:
            before = watermark.get_watermark(session, FEED).last_slice_key
            # Mark an old window terminal without touching the watermark.
            for key in ["20260101000000", "20260101001500"]:
                set_slice_status(session, key, "ok")
            after = watermark.get_watermark(session, FEED).last_slice_key

        assert before == after == "20260826120000"


class TestLag:
    def test_lag_reflects_the_watermark(self):
        with session_scope() as session:
            watermark.advance_watermark(session, gdelt.slice_key(datetime.now(UTC)), FEED)
            info = catchup.status(session, feed=FEED)
            assert info["lag_slices"] == 0

    def test_lag_is_none_before_any_ingest(self):
        with session_scope() as session:
            info = catchup.status(session, feed=FEED)
            assert info["watermark"] is None
            assert info["lag_slices"] is None


class TestAdvisoryLock:
    """Advisory locks are scoped to a *connection*, so these need real ones.

    The per-test rollback fixture binds every session to one shared connection,
    which would let the same connection take the lock twice and make the test
    pass for the wrong reason. These open their own engines instead.
    """

    @staticmethod
    def _sessions(url: str):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(url, poolclass=NullPool, future=True)
        return engine, sessionmaker(bind=engine, future=True)

    def test_lock_is_exclusive_across_connections(self, test_database):
        """Two watchers, or a watcher racing a manual run, must not both fetch."""
        key = 0x484F4E55  # a test-only key, distinct from the real one
        engine_a, factory_a = self._sessions(test_database)
        engine_b, factory_b = self._sessions(test_database)
        try:
            with factory_a() as first:
                assert try_lock(first, key) is True
                with factory_b() as second:
                    assert try_lock(second, key) is False
                unlock(first, key)
        finally:
            engine_a.dispose()
            engine_b.dispose()

    def test_lock_is_reacquirable_after_release(self, test_database):
        key = 0x484F4E56
        engine, factory = self._sessions(test_database)
        try:
            with factory() as session:
                assert try_lock(session, key) is True
                unlock(session, key)
            with factory() as session:
                assert try_lock(session, key) is True
                unlock(session, key)
        finally:
            engine.dispose()
