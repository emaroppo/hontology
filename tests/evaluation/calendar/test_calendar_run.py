"""When a calendar window counts as ingested."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from hontology.db.models import FeedSlice
from hontology.db.session import session_scope
from hontology.evaluation.calendar.events import Entry
from hontology.evaluation.calendar.runner import (
    SLICES_PER_DAY,
    window_status,
    window_unobservable,
)
from hontology.pipeline.ingest.feed import slices

pytestmark = pytest.mark.requires_db

ENTRY = Entry("e", "positive", "Port closure", ("HKG",), date(2025, 9, 23), None, "")
PLACE = 7


def fill(*, gkg_scope: list[int] | None, skip: int = 0, fail: int = 0) -> None:
    """Ingest-status rows for the entry's one-day window (before=0, after=0)."""
    start = datetime(2025, 9, 23, tzinfo=UTC)
    with session_scope() as session:
        for feed, scope in ((slices.FEED, None), (slices.FEED_GKG, gkg_scope)):
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


def test_a_graph_slice_kept_for_another_country_is_ingested_again():
    """Left alone, nothing would widen its scope and the window would wait
    forever; it is handed back for ingesting with this entry's places."""
    fill(gkg_scope=[PLACE + 1])
    ready, again = status()
    assert ready is False
    assert len(again) == SLICES_PER_DAY
    assert {feed for feed, _ in again} == {slices.FEED_GKG}


def test_a_full_graph_slice_counts():
    fill(gkg_scope=None)
    assert status()[0] is True


def unobservable() -> bool:
    with session_scope() as session:
        return window_unobservable(session, ENTRY, [PLACE], before=0, after=0)


def mark(status: str, feed: str | None = None) -> None:
    with session_scope() as session:
        for row in session.scalars(select(FeedSlice)):
            if feed is None or row.feed == feed:
                row.status = status


def test_a_window_the_source_never_published_is_unobservable():
    """Every slice missing in both feeds: ingested, but nothing to see."""
    fill(gkg_scope=[PLACE])
    mark("missing")
    assert status()[0] is True
    assert unobservable() is True


def test_one_feed_published_is_enough_to_observe():
    fill(gkg_scope=[PLACE])
    mark("missing", feed=slices.FEED_GKG)
    assert unobservable() is False


def test_a_graph_slice_kept_for_another_country_is_no_observation():
    fill(gkg_scope=[PLACE + 1])
    mark("missing", feed=slices.FEED)
    assert unobservable() is True


def test_an_ingested_window_is_observable():
    fill(gkg_scope=[PLACE])
    assert unobservable() is False
