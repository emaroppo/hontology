"""Storing one downloaded slice: its documents, and its events or articles.

Documents upsert on their URL and feed rows on ``(slice, record)``, so storing a
slice again changes nothing.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from hontology.db.lookups import count
from hontology.db.models import Document, FeedArticle, FeedEvent, FeedSlice
from hontology.pipeline.ingest.codes.loci import by_fips
from hontology.pipeline.ingest.feed import gdelt, parse


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def get_or_create_slice(session: Session, key: str, feed: str) -> FeedSlice:
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


def upsert_documents(session: Session, urls: list[str]) -> dict[str, int]:
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


def store_export(
    session: Session, row: FeedSlice, payload: bytes
) -> tuple[int, int, dict[str, int]]:
    fips_lookup = by_fips(session)
    events = list(parse.parse_export(payload))
    doc_ids = upsert_documents(session, [record["url"] for record in events])

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


def store_gkg(
    session: Session, row: FeedSlice, payload: bytes, wanted: list[int] | None
) -> tuple[int, int, dict[str, int]]:
    fips_lookup = by_fips(session)
    keep = set(wanted) if wanted is not None else None
    records: list[dict] = []
    total = 0

    for record in parse.parse_gkg(payload):
        total += 1
        locus_ids = sorted(
            {fips_lookup[code].id for code in record["countries"] if code in fips_lookup}
        )
        if keep is not None and keep.isdisjoint(locus_ids):
            continue
        records.append(record | {"locus_ids": locus_ids})

    doc_ids = upsert_documents(session, [record["url"] for record in records])
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


def count_unfetched(session: Session, doc_ids: list[int]) -> int:
    if not doc_ids:
        return 0
    return count(session, Document.id, Document.id.in_(doc_ids), Document.fetched_at.is_(None))
