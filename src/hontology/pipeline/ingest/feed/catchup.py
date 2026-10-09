"""Catching up to the newest published slice, backfilling a window, and
reporting where ingest stands."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.lookups import count
from hontology.db.models import Document, FeedSlice
from hontology.pipeline.ingest.feed import gdelt, watermark
from hontology.pipeline.ingest.feed.slices import FEED, TERMINAL, feed_kind, ingest_slice


def catch_up(session: Session, *, max_slices: int = 32, feed: str = FEED) -> dict:
    """Process every slice between the watermark and the newest published one.

    Bounded by *max_slices* so a long outage becomes several ordered passes
    rather than one unbounded crawl. The watermark advances only across a
    contiguous run, so an interrupted catch-up resumes exactly where it stopped.
    """
    settings = get_settings()
    latest = gdelt.fetch_lastupdate(settings.gdelt_base_url, kind=feed_kind(feed))
    mark = watermark.get_watermark(session, feed)

    if mark.last_slice_key is None:
        # A fresh install starts at the newest slice rather than at the feed's
        # beginning: backfilling years of history is an explicit decision, not
        # something an empty database should do on its own.
        pending = [latest]
    else:
        pending = gdelt.keys_between(mark.last_slice_key, latest, limit=max_slices)

    results = []
    for key in pending:
        result = ingest_slice(session, key, feed=feed)
        results.append(result)
        if result["status"] in TERMINAL:
            watermark.advance_watermark(session, key, feed)
        else:
            # Leave the watermark behind an unfinished slice so the next pass
            # retries it instead of stepping over it.
            break

    return {
        "latest_published": latest,
        "watermark": watermark.get_watermark(session, feed).last_slice_key,
        "processed": len(results),
        "remaining": max(
            0,
            len(
                gdelt.keys_between(
                    watermark.get_watermark(session, feed).last_slice_key or latest, latest
                )
            ),
        ),
        "slices": results,
    }


def backfill(
    session: Session, start: str, end: str, *, feed: str = FEED, loci: list[int] | None = None
) -> dict:
    """Ingest an explicit historical window.

    Deliberately does **not** touch the watermark: a backfill of last month must
    not convince the scheduler it is caught up to now and skip the live gap.
    """
    results = [
        ingest_slice(session, key, feed=feed, loci=loci)
        for key in [start, *gdelt.keys_between(start, end)]
    ]
    return {"start": start, "end": end, "processed": len(results), "slices": results}


def status(session: Session, *, feed: str = FEED) -> dict:
    """What an operator needs to know: are we keeping up, and where are the gaps."""
    mark = watermark.get_watermark(session, feed)
    now = datetime.now(UTC)

    counts: dict[str, int] = {
        str(status): int(n)
        for status, n in session.execute(
            select(FeedSlice.status, func.count(FeedSlice.id))
            .where(FeedSlice.feed == feed)
            .group_by(FeedSlice.status)
        ).all()
    }
    return {
        "feed": feed,
        "watermark": mark.last_slice_key,
        "lag_slices": gdelt.lag_slices(mark.last_slice_key, now),
        "slice_counts": counts,
        "documents": count(session, Document.id),
        "documents_unfetched": count(session, Document.id, Document.fetched_at.is_(None)),
    }
