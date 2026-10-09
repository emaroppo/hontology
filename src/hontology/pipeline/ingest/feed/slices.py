"""Slice ingestion, catch-up and backfill.

The unit of work is one slice, and processing it is **idempotent**: the stamp is
the identity, documents upsert on their URL, and feed events upsert on
``(slice, event id)``. Re-running a slice therefore costs a download and changes
nothing, which is what makes an unattended restart safe.

The watermark (`pipeline.ingest.feed.watermark`) moves only onto a slice that has
reached a terminal state, so a slice that failed transport is retried on the next
pass. A slice the feed never published is terminal too — otherwise one permanent
gap stalls the ingester forever. Storing a downloaded slice is in
`pipeline.ingest.feed.store`.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import FeedSlice
from hontology.pipeline.ingest.feed import gdelt, store

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


# ---------------------------------------------------------------------------
# One slice
# ---------------------------------------------------------------------------


def _covers(done: list[int] | None, wanted: list[int] | None) -> bool:
    """Whether a slice processed with scope *done* already holds scope *wanted*.

    ``None`` is everything: a full slice covers any request, and a full request
    is covered only by a full slice.
    """
    if done is None:
        return True
    return wanted is not None and set(wanted) <= set(done)


def _download(session: Session, row: FeedSlice, key: str, kind: str) -> bytes | dict:
    """The slice's file, or, when there is none, the result recorded on *row*."""
    try:
        return gdelt.fetch_slice(get_settings().gdelt_base_url, key, kind=kind)
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
    started = time.monotonic()
    row = store.get_or_create_slice(session, key, feed)
    kind = feed_kind(feed)
    wanted = sorted(set(loci)) if loci is not None and kind == gdelt.KIND_GKG else None

    if row.status in TERMINAL and not force and _covers(row.scope, wanted):
        return {"slice_key": key, "status": row.status, "skipped": True}

    payload = _download(session, row, key, kind)
    if isinstance(payload, dict):
        return payload

    if kind == gdelt.KIND_GKG:
        rows_in_feed, kept, doc_ids = store.store_gkg(session, row, payload, wanted)
        # Widen rather than replace: records kept for an earlier scope are
        # still in the table.
        previous = row.scope if row.status in TERMINAL else []
        row.scope = (
            None if wanted is None or previous is None else sorted(set(previous) | set(wanted))
        )
    else:
        rows_in_feed, kept, doc_ids = store.store_export(session, row, payload)

    new_documents = store.count_unfetched(session, list(doc_ids.values()))
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
