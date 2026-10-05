"""When a calendar window counts as ingested."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from hontology.db.models import FeedSlice
from hontology.db.session import session_scope
from hontology.evalkit.calendar import Entry
from hontology.evalkit.calendar_run import SLICES_PER_DAY, window_status
from hontology.ingest import service

pytestmark = pytest.mark.requires_db

ENTRY = Entry("e", "positive", "Port closure", ("HKG",), date(2025, 9, 23), None, "")
PLACE = 7


def fill(*, gkg_scope: list[int] | None, skip: int = 0, fail: int = 0) -> None:
    """Ingest-status rows for the entry's one-day window (before=0, after=0)."""
    start = datetime(2025, 9, 23, tzinfo=UTC)
    with session_scope() as session:
        for feed, scope in ((service.FEED, None), (service.FEED_GKG, gkg_scope)):
            for i in range(SLICES_PER_DAY - skip):
                moment = start + timedelta(minutes=15 * i)
                session.add(
                    FeedSlice(
                        feed=feed,
                        slice_key=moment.strftime("%Y%m%d%H%M%S"),
                        sliced_at=moment,
                        status="failed" if i < fail else "ok",
                        scope=scope,
                    )
                )


def status():
    with session_scope() as session:
        return window_status(session, ENTRY, [PLACE], before=0, after=0)


def test_every_slice_in_both_feeds_makes_it_ready():
    fill(gkg_scope=[PLACE])
    assert status() == (True, [])


def test_a_missing_slice_keeps_it_waiting():
    fill(gkg_scope=[PLACE], skip=1)
    assert status()[0] is False


def test_failed_slices_are_reported_for_retry():
    fill(gkg_scope=[PLACE], fail=2)
    ready, failed = status()
    assert ready is False
    assert len(failed) == 4  # two per feed


def test_a_graph_slice_kept_for_another_country_does_not_count():
    fill(gkg_scope=[PLACE + 1])
    assert status()[0] is False


def test_a_full_graph_slice_counts():
    fill(gkg_scope=None)
    assert status()[0] is True
