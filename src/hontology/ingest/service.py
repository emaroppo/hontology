"""Slice ingestion, catch-up and backfill.

The unit of work is one slice, and processing it is **idempotent**: the stamp is
the identity, documents upsert on their URL, and feed events upsert on
``(slice, event id)``. Re-running a slice therefore costs a download and changes
nothing, which is what makes an unattended restart safe.

The watermark advances **contiguously**. It moves to a slice only when that slice
is the immediate successor of the current watermark and has reached a terminal
state, so a slice that failed transport leaves the watermark behind it and gets
retried on the next pass. A slice the feed never published is terminal too —
otherwise one permanent gap stalls the ingester forever.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Document, FeedArticle, FeedEvent, FeedSlice, IngestWatermark
from hontology.ingest import gdelt
from hontology.ingest.loci import by_fips

log = logging.getLogger(__name__)

FEED = "gdelt_v2"
# The Global Knowledge Graph: every article GDELT read, with themes and places.
FEED_GKG = "gdelt_gkg"
FEED_KINDS = {FEED: gdelt.KIND_EXPORT, FEED_GKG: gdelt.KIND_GKG}


def feed_kind(feed: str) -> str:
    """Which file a feed reads. Any feed not named for the GKG reads the export."""
    return FEED_KINDS.get(feed, gdelt.KIND_EXPORT)


# Statuses a slice can end in. `pending` is the only non-terminal one.
TERMINAL = {"ok", "empty", "missing"}

# How long to keep retrying a stamp the feed has not published before accepting
# that it never will. GDELT skips slices; without this the ingester would retry
# one gap forever and never advance.
MISSING_AFTER_SECONDS = 3 * 3600


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


def get_watermark(session: Session, feed: str = FEED) -> IngestWatermark:
    row = session.scalar(select(IngestWatermark).where(IngestWatermark.feed == feed))
    if row is None:
        row = IngestWatermark(feed=feed)
        session.add(row)
        session.flush()
    return row


def advance_watermark(session: Session, key: str, feed: str = FEED) -> bool:
    """Move the watermark to *key* if it is the next contiguous slice.

    Returns whether it moved. Refusing to skip ahead is what stops a successful
    later slice from marking an earlier failed one as done.
    """
    mark = get_watermark(session, feed)
    if mark.last_slice_key is not None and key != gdelt.next_key(mark.last_slice_key):
        return False
    mark.last_slice_key = key
    mark.last_sliced_at = gdelt.key_to_datetime(key)
    session.flush()
    return True


# ---------------------------------------------------------------------------
# One slice
# ---------------------------------------------------------------------------


def _get_or_create_slice(session: Session, key: str, feed: str = FEED) -> FeedSlice:
    row = session.scalar(
        select(FeedSlice).where(FeedSlice.feed == feed, FeedSlice.slice_key == key)
    )
    if row is None:
        row = FeedSlice(
            feed=feed, slice_key=key, sliced_at=gdelt.key_to_datetime(key), status="pending"
        )
        session.add(row)
        session.flush()
    return row


def _upsert_documents(session: Session, urls: list[str]) -> dict[str, int]:
    """Insert any unseen URLs as un-fetched documents; return ``{url: id}``.

    Documents are global, not per-slice: the same article cited by two slices is
    one row and gets scraped once ever.
    """
    if not urls:
        return {}

    # Sorted so concurrent ingesters (a watcher and a backfill, or parallel
    # backfills) take row locks in the same order and cannot deadlock.
    rows = [{"url": url, "url_hash": url_hash(url)} for url in sorted(set(urls))]
    session.execute(
        pg_insert(Document).on_conflict_do_nothing(index_elements=["url"]),
        rows,
    )
    session.flush()

    found = session.execute(select(Document.url, Document.id).where(Document.url.in_(urls)))
    return {url: doc_id for url, doc_id in found}


def _covers(done: list[int] | None, wanted: list[int] | None) -> bool:
    """Whether a slice processed with scope *done* already holds scope *wanted*.

    ``None`` is everything: a full slice covers any request, and a full request
    is covered only by a full slice.
    """
    if done is None:
        return True
    return wanted is not None and set(wanted) <= set(done)


def ingest_slice(
    session: Session,
    key: str,
    *,
    feed: str = FEED,
    force: bool = False,
    loci: list[int] | None = None,
) -> dict:
    """Fetch and store one slice. Safe to call repeatedly.

    *loci* restricts a GKG slice to articles mentioning one of those places. The
    export is always kept whole: it is small, and its rows are what the code
    filter reads.
    """
    settings = get_settings()
    started = time.monotonic()
    row = _get_or_create_slice(session, key, feed)
    kind = feed_kind(feed)
    wanted = sorted(set(loci)) if loci is not None and kind == gdelt.KIND_GKG else None

    if row.status in TERMINAL and not force and _covers(row.scope, wanted):
        return {"slice_key": key, "status": row.status, "skipped": True}

    try:
        payload = gdelt.fetch_slice(settings.gdelt_base_url, key, kind=kind)
    except gdelt.SliceNotPublished:
        age = (datetime.now(UTC) - gdelt.key_to_datetime(key)).total_seconds()
        # Not yet published is normal; never published is a real gap.
        row.status = "missing" if age > MISSING_AFTER_SECONDS else "pending"
        row.error = "not published by the feed"
        session.flush()
        return {"slice_key": key, "status": row.status, "skipped": False}
    except httpx.HTTPError as exc:
        row.status = "failed"
        row.error = str(exc)[:500]
        session.flush()
        log.warning("slice %s failed: %s", key, exc)
        return {"slice_key": key, "status": "failed", "error": str(exc)}

    if kind == gdelt.KIND_GKG:
        rows_in_feed, kept, doc_ids = _store_gkg(session, row, payload, wanted)
        # Widen rather than replace: records kept for an earlier scope are
        # still in the table.
        previous = row.scope if row.status in TERMINAL else []
        row.scope = (
            None if wanted is None or previous is None else sorted(set(previous) | set(wanted))
        )
    else:
        rows_in_feed, kept, doc_ids = _store_export(session, row, payload)

    new_documents = _count_unfetched(session, list(doc_ids.values()))
    row.rows_in_feed = rows_in_feed
    row.rows_matched = kept
    row.documents_new = new_documents
    row.documents_cached = len(doc_ids) - new_documents
    row.duration_s = time.monotonic() - started
    row.status = "ok" if rows_in_feed else "empty"
    row.error = None
    session.flush()

    log.info(
        "slice %s %s: %s rows, %s kept, %s documents (%s new) in %.1fs",
        feed,
        key,
        rows_in_feed,
        kept,
        len(doc_ids),
        new_documents,
        row.duration_s,
    )
    return {
        "slice_key": key,
        "status": row.status,
        "rows": rows_in_feed,
        "kept": kept,
        "documents": len(doc_ids),
        "documents_new": new_documents,
        "duration_s": row.duration_s,
    }


def _store_export(
    session: Session, row: FeedSlice, payload: bytes
) -> tuple[int, int, dict[str, int]]:
    fips_lookup = by_fips(session)
    events = list(gdelt.parse_export(payload))
    doc_ids = _upsert_documents(session, [record["url"] for record in events])

    if events:
        session.execute(
            pg_insert(FeedEvent).on_conflict_do_nothing(
                index_elements=["slice_id", "feed_event_id"]
            ),
            [
                {
                    "slice_id": row.id,
                    "feed_event_id": record["event_id"],
                    "document_id": doc_ids.get(record["url"]),
                    "event_code": record["event_code"] or None,
                    "base_code": record["base_code"] or None,
                    "root_code": record["root_code"] or None,
                    "locus_id": (
                        fips_lookup[record["country"]].id
                        if record["country"] in fips_lookup
                        else None
                    ),
                    "occurred_on": record["day"] or None,
                }
                for record in events
            ],
        )
    return len(events), len(events), doc_ids


def _store_gkg(
    session: Session, row: FeedSlice, payload: bytes, wanted: list[int] | None
) -> tuple[int, int, dict[str, int]]:
    fips_lookup = by_fips(session)
    keep = set(wanted) if wanted is not None else None
    records: list[dict] = []
    total = 0

    for record in gdelt.parse_gkg(payload):
        total += 1
        locus_ids = sorted(
            {fips_lookup[code].id for code in record["countries"] if code in fips_lookup}
        )
        if keep is not None and keep.isdisjoint(locus_ids):
            continue
        records.append(record | {"locus_ids": locus_ids})

    doc_ids = _upsert_documents(session, [record["url"] for record in records])
    if records:
        session.execute(
            pg_insert(FeedArticle).on_conflict_do_nothing(
                index_elements=["slice_id", "record_id"]
            ),
            [
                {
                    "slice_id": row.id,
                    "record_id": record["record_id"],
                    "document_id": doc_ids.get(record["url"]),
                    "published_at": (
                        gdelt.key_to_datetime(record["date"]) if record["date"] else None
                    ),
                    "themes": sorted(record["themes"]),
                    "locus_ids": record["locus_ids"],
                }
                for record in records
            ],
        )
    return total, len(records), doc_ids


def _count_unfetched(session: Session, doc_ids: list[int]) -> int:
    if not doc_ids:
        return 0
    return (
        session.scalar(
            select(func.count(Document.id)).where(
                Document.id.in_(doc_ids), Document.fetched_at.is_(None)
            )
        )
        or 0
    )


# ---------------------------------------------------------------------------
# Catch-up and backfill
# ---------------------------------------------------------------------------


def catch_up(session: Session, *, max_slices: int = 32, feed: str = FEED) -> dict:
    """Process every slice between the watermark and the newest published one.

    Bounded by *max_slices* so a long outage becomes several ordered passes
    rather than one unbounded crawl. The watermark advances only across a
    contiguous run, so an interrupted catch-up resumes exactly where it stopped.
    """
    settings = get_settings()
    latest = gdelt.fetch_lastupdate(settings.gdelt_base_url, kind=feed_kind(feed))
    mark = get_watermark(session, feed)

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
            advance_watermark(session, key, feed)
        else:
            # Leave the watermark behind an unfinished slice so the next pass
            # retries it instead of stepping over it.
            break

    return {
        "latest_published": latest,
        "watermark": get_watermark(session, feed).last_slice_key,
        "processed": len(results),
        "remaining": max(
            0,
            len(
                gdelt.keys_between(
                    get_watermark(session, feed).last_slice_key or latest, latest
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
    mark = get_watermark(session, feed)
    now = datetime.now(UTC)

    counts: dict[str, int] = {
        str(status): int(count)
        for status, count in session.execute(
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
        "documents": session.scalar(select(func.count(Document.id))) or 0,
        "documents_unfetched": session.scalar(
            select(func.count(Document.id)).where(Document.fetched_at.is_(None))
        )
        or 0,
    }
