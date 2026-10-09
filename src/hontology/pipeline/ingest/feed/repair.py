"""Giving old event rows their country, by re-reading their export slices."""

from __future__ import annotations

from sqlalchemy import bindparam, select, update
from sqlalchemy.orm import Session

from hontology.db.models import FeedEvent, FeedSlice
from hontology.pipeline.ingest.feed import parse
from hontology.pipeline.ingest.feed.slices import FEED


def slices_missing_export_loci(session: Session, *, feed: str = FEED) -> list[str]:
    """Export slices none of whose events carry a country.

    Before the column fix every event was stored without one; a slice ingested
    since has at least one located event, so this finds exactly the old ones.
    """
    located = (
        select(FeedEvent.slice_id).where(FeedEvent.locus_id.is_not(None)).distinct().subquery()
    )
    return list(
        session.scalars(
            select(FeedSlice.slice_key)
            .where(
                FeedSlice.feed == feed,
                FeedSlice.status == "ok",
                FeedSlice.id.not_in(select(located.c.slice_id)),
            )
            .order_by(FeedSlice.slice_key)
        )
    )


def export_loci(payload: bytes, fips_lookup: dict) -> dict[str, int]:
    """``{event id: locus id}`` for the events in a downloaded export slice."""
    return {
        record["event_id"]: fips_lookup[record["country"]].id
        for record in parse.parse_export(payload)
        if record["country"] in fips_lookup
    }


def apply_export_loci(
    session: Session, key: str, loci_by_event: dict[str, int], *, feed: str = FEED
) -> int:
    """Set the country of a slice's existing events. Returns rows updated."""
    slice_id = session.scalar(
        select(FeedSlice.id).where(FeedSlice.feed == feed, FeedSlice.slice_key == key)
    )
    if slice_id is None or not loci_by_event:
        return 0
    # A Core statement on the plain connection, so the rows go as one
    # executemany rather than one ORM round trip per event. Parameter names must
    # not repeat column names, hence the prefix.
    statement = (
        update(FeedEvent)
        .where(FeedEvent.slice_id == slice_id, FeedEvent.feed_event_id == bindparam("b_event"))
        .values(locus_id=bindparam("b_locus"))
    )
    session.connection().execute(
        statement,
        [{"b_event": event_id, "b_locus": locus} for event_id, locus in loci_by_event.items()],
    )
    return len(loci_by_event)
