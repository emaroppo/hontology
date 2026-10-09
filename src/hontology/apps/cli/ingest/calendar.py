"""`hontology ingest` for an event calendar: backfill its windows, preview their size."""

from __future__ import annotations

from pathlib import Path

import typer
from sqlalchemy import func, select

from hontology.apps.cli.common import After, Before, log_to_console
from hontology.db.session import session_scope
from hontology.pipeline.ingest.feed import gdelt, slices


def ingest_calendar(calendar_path: Path, before: Before = 1, after: After = 2) -> None:
    """Backfill both feeds for every window an event calendar needs.

    The event export is kept whole; the knowledge graph, roughly seventy times
    larger, keeps only articles mentioning a country the calendar names on that
    day. Each slice commits on its own, so an interrupted backfill resumes where
    it stopped.
    """
    from hontology.evaluation.calendar import events as calendar

    log_to_console()
    entries = calendar.load(calendar_path)
    with session_scope() as session:
        loci = calendar.loci_for(session, entries)
    days = calendar.days_to_ingest(entries, loci, before=before, after=after)
    typer.echo(f"{len(days)} day(s), {len(days) * 96} slice(s) per feed")

    totals: dict[str, int] = {}
    for index, (day, places) in enumerate(days.items(), start=1):
        start = day.strftime("%Y%m%d") + "000000"
        for key in [start, *gdelt.keys_between(start, day.strftime("%Y%m%d") + "234500")]:
            for feed, scope in ((slices.FEED, None), (slices.FEED_GKG, places)):
                with session_scope() as session:
                    result = slices.ingest_slice(session, key, feed=feed, loci=scope)
                status_key = "skipped" if result.get("skipped") else result["status"]
                totals[status_key] = totals.get(status_key, 0) + 1
        typer.echo(f"[{index}/{len(days)}] {day}  {totals}")


def ingest_calendar_preview(
    calendar_path: Path,
    ontology_id: int | None = typer.Option(None, help="Also count what the filter keeps."),
    before: Before = 1,
    after: After = 2,
) -> None:
    """How many documents each calendar window holds, before fetching any."""
    from hontology.db.models import Document
    from hontology.evaluation.calendar import events as calendar
    from hontology.pipeline.ingest.articles import filter as ingest_filter

    entries = calendar.load(calendar_path)
    with session_scope() as session:
        loci = calendar.loci_for(session, entries)
        passed = (
            set(ingest_filter.matching_documents(session, ontology_id)) if ontology_id else None
        )
        everything: set[int] = set()
        kept: set[int] = set()
        typer.echo(f"{'entry':40} {'in feed':>8} {'filter':>8}")
        for entry in entries:
            docs = set(
                calendar.window_documents(session, entry, loci, before=before, after=after)
            )
            everything |= docs
            hit = docs & passed if passed is not None else docs
            kept |= hit
            shown = len(hit) if passed is not None else "-"
            typer.echo(f"{entry.id:40} {len(docs):>8} {shown:>8}")
        unfetched = session.scalar(
            select(func.count()).where(Document.id.in_(kept), Document.fetched_at.is_(None))
        )
    typer.echo(
        f"distinct documents: {len(everything)} in windows, {len(kept)} kept, "
        f"{unfetched} still to fetch"
    )
