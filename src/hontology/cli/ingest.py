"""`hontology ingest`: pull slices from the news feed, then fetch and tidy articles.

One-shot commands are the baseline. `watch` is the only continuous one, and it
has to be started deliberately — nothing else in the system spawns it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from sqlalchemy import func, select

from hontology.cli.common import After, Before, calendar_documents, log_to_console
from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.ingest import gdelt, scrape, service

app = typer.Typer(help="Pull slices from the news feed.")
# Documents fetched per committed batch by `ingest scrape`.
SCRAPE_BATCH = 200
CalendarScope = typer.Option(
    None, "--calendar", help="Only documents inside this calendar's windows."
)


@app.command("status")
def ingest_status() -> None:
    """Show the watermark, how far behind it is, and where the gaps are."""
    with session_scope() as session:
        info = service.status(session)

    lag = info["lag_slices"]
    typer.echo(f"watermark    {info['watermark'] or '(never ingested)'}")
    if lag is None:
        typer.echo("lag          n/a — nothing ingested yet")
    else:
        typer.secho(
            f"lag          {lag} slice(s) ≈ {lag * 15} min",
            fg=typer.colors.GREEN if lag <= 2 else typer.colors.YELLOW,
        )
    typer.echo(f"slices       {info['slice_counts'] or '(none)'}")
    typer.echo(
        f"documents    {info['documents']} ({info['documents_unfetched']} not yet fetched)"
    )


@app.command("once")
def ingest_once(
    max_slices: int = typer.Option(32, help="Upper bound on slices for this pass."),
) -> None:
    """Catch up to the newest published slice, then exit."""
    with session_scope() as session:
        result = service.catch_up(session, max_slices=max_slices)
    typer.echo(
        f"latest={result['latest_published']} watermark={result['watermark']} "
        f"processed={result['processed']} remaining={result['remaining']}"
    )
    for row in result["slices"]:
        typer.echo(f"  {row['slice_key']}  {row['status']:<8} {row.get('rows', '')}")


@app.command("backfill")
def ingest_backfill(start: str, end: str) -> None:
    """Ingest an explicit window. Does not move the watermark."""
    with session_scope() as session:
        result = service.backfill(session, start, end)
    typer.echo(f"processed {result['processed']} slice(s) from {start} to {end}")


@app.command("themes")
def ingest_themes() -> None:
    """Load the GKG theme vocabulary as a code system, for theme links."""
    from hontology.ingest import themes

    with session_scope() as session:
        result = themes.ingest(session)
    typer.echo(f"gkg themes: {result['inserted']} new, {result['total']} total")


@app.command("calendar")
def ingest_calendar(calendar_path: Path, before: Before = 1, after: After = 2) -> None:
    """Backfill both feeds for every window an event calendar needs.

    The event export is kept whole; the knowledge graph, roughly seventy times
    larger, keeps only articles mentioning a country the calendar names on that
    day. Each slice commits on its own, so an interrupted backfill resumes where
    it stopped.
    """
    from hontology.evalkit import calendar

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
            for feed, scope in ((service.FEED, None), (service.FEED_GKG, places)):
                with session_scope() as session:
                    result = service.ingest_slice(session, key, feed=feed, loci=scope)
                status_key = "skipped" if result.get("skipped") else result["status"]
                totals[status_key] = totals.get(status_key, 0) + 1
        typer.echo(f"[{index}/{len(days)}] {day}  {totals}")


@app.command("repair-export-loci")
def ingest_repair_export_loci(
    workers: int = typer.Option(8, help="Slices downloaded in parallel."),
    limit: int | None = typer.Option(None, help="Repair at most this many slices."),
) -> None:
    """Give old event rows their country, by re-reading their export slices.

    Events ingested before the column fix were stored without a country. Each
    slice is downloaded again, read with the corrected parser and its rows
    updated in place, committing per slice, so the repair can stop and resume.
    """
    from concurrent.futures import ThreadPoolExecutor

    from hontology.ingest.loci import by_fips

    settings = get_settings()
    with session_scope() as session:
        keys = service.slices_missing_export_loci(session)
        lookup = by_fips(session)
    keys = keys[:limit] if limit is not None else keys
    typer.echo(f"{len(keys)} slice(s) to repair")

    def fetch(key: str) -> tuple[str, dict[str, int] | None]:
        try:
            payload = gdelt.fetch_slice(settings.gdelt_base_url, key)
        except Exception:  # noqa: BLE001 - a failed slice is reported, not fatal
            return key, None
        return key, service.export_loci(payload, lookup)

    updated = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (key, loci) in enumerate(pool.map(fetch, keys), start=1):
            if loci is None:
                failed += 1
            else:
                with session_scope() as session:
                    updated += service.apply_export_loci(session, key, loci)
            if done % 500 == 0 or done == len(keys):
                typer.echo(f"[{done}/{len(keys)}] {updated} event(s) located, {failed} failed")


@app.command("calendar-preview")
def ingest_calendar_preview(
    calendar_path: Path,
    ontology_id: int | None = typer.Option(None, help="Also count what the filter keeps."),
    before: Before = 1,
    after: After = 2,
) -> None:
    """How many documents each calendar window holds, before fetching any."""
    from hontology.db.models import Document
    from hontology.evalkit import calendar
    from hontology.ingest import filter as ingest_filter

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


@app.command("scrape")
def ingest_scrape(
    limit: int | None = typer.Option(None, help="Max documents this run (default: budget)."),
    retry_failed: bool = typer.Option(
        False, help="Also re-attempt URLs already recorded as failures."
    ),
    reader_proxy: bool = typer.Option(
        False, help="Allow the third-party reader service as a last resort."
    ),
    ontology_id: int | None = typer.Option(
        None,
        help="Only fetch documents whose feed codes reach this ontology's concepts. "
        "Off by default; meaningless for an ontology with no code links.",
    ),
    calendar_path: Path | None = CalendarScope,
    before: Before = 1,
    after: After = 2,
) -> None:
    """Fetch article text for documents that do not have it yet.

    Work commits in batches, so a long scrape that is stopped or crashes keeps
    everything but its last batch instead of losing the whole run.
    """
    from hontology.ingest import filter as ingest_filter

    log_to_console()
    with session_scope() as session:
        document_ids = None
        if calendar_path is not None:
            document_ids = calendar_documents(session, calendar_path, before, after)
        if ontology_id is not None:
            # Resolved once here rather than per batch: the answer does not
            # change while fetching. With a calendar, only its windows are
            # matched; matching every feed record loads all of them.
            matches = set(
                ingest_filter.matching_documents(
                    session,
                    ontology_id,
                    document_ids=sorted(document_ids) if document_ids is not None else None,
                )
            )
            if not matches:
                typer.secho(
                    "no documents match this ontology's code links; nothing to fetch",
                    fg=typer.colors.YELLOW,
                )
                raise typer.Exit(1)
            document_ids = matches if document_ids is None else document_ids & matches
    allowed = sorted(document_ids) if document_ids is not None else None
    if allowed is not None:
        typer.echo(f"{len(allowed)} document(s) in scope")

    budget = limit if limit is not None else get_settings().scrape_budget

    def scrape_batch(held: list[int]) -> dict:
        with session_scope() as session:
            return scrape.scrape_pending(
                session,
                limit=min(SCRAPE_BATCH, budget),
                retry_failed=retry_failed,
                use_reader_proxy=reader_proxy,
                document_ids=allowed,
                exclude=held,
            )

    def report(totals: dict[str, int]) -> None:
        typer.echo(
            f"attempted {totals['attempted']}: {totals['ok']} ok, {totals['junk']} junk, "
            f"{totals['failed']} failed, {totals['blocked_by_robots']} blocked by robots, "
            f"{totals['deferred']} deferred by crawl delay, "
            f"{totals['retry_later']} left for a later batch after a connection failure"
        )

    if retry_failed:
        # A retry pass would re-attempt the same failures every batch; one is enough.
        first = scrape_batch([])
        totals = {key: first.get(key, 0) for key in scrape.TOTAL_KEYS}
        if totals["attempted"]:
            report(totals)
    else:
        totals = scrape.drain(scrape_batch, budget=budget, on_batch=report)
    if not totals["attempted"]:
        typer.echo("nothing to fetch")


@app.command("dedup")
def ingest_dedup(
    calendar_path: Path | None = CalendarScope,
    before: Before = 1,
    after: After = 2,
    threshold: float = typer.Option(0.8, help="Estimated Jaccard similarity to group at."),
) -> None:
    """Group near-duplicate articles so only one of each is retrieved and judged."""
    from hontology.db.models import Document
    from hontology.ingest import dedup

    with session_scope() as session:
        if calendar_path is not None:
            ids = calendar_documents(session, calendar_path, before, after)
        else:
            ids = set(
                session.scalars(select(Document.id).where(Document.body_path.is_not(None)))
            )
        result = dedup.deduplicate(session, sorted(ids), threshold=threshold)
    copies = result["documents"] - result["representatives"]
    typer.echo(
        f"{result['documents']} fetched document(s): {result['representatives']} to judge, "
        f"{result['newly_marked']} newly marked as copies "
        f"({result['too_short']} too short to fingerprint)"
    )
    if result["documents"]:
        typer.echo(f"judging load cut by {copies / result['documents']:.0%}")


@app.command("filter-preview")
def ingest_filter_preview(ontology_id: int) -> None:
    """Show what the pre-scrape code filter would keep, without fetching."""
    from hontology.ingest import filter as ingest_filter

    with session_scope() as session:
        report = ingest_filter.preview(session, ontology_id)

    if not report["usable"]:
        typer.secho(report["reason"], fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    typer.echo(f"linked codes        {report['linked_codes']}")
    typer.echo(f"documents total     {report['documents_total']}")
    typer.echo(f"  matching          {report['documents_matching']}")
    typer.echo(f"unfetched           {report['documents_unfetched']}")
    typer.echo(f"  would fetch       {report['unfetched_matching']}")
    typer.echo(f"  would skip        {report['unfetched_skipped']}")
    share = report["share_kept"]
    if share is not None:
        typer.secho(f"share kept          {share:.1%}", fg=typer.colors.GREEN)


@app.command("watch")
def ingest_watch(
    grace_seconds: float = typer.Option(
        90.0, help="Delay after each quarter-hour boundary before polling."
    ),
    max_slices: int = typer.Option(32, help="Upper bound on slices per pass."),
) -> None:
    """Run continuously, following the feed's 15-minute cadence.

    Opt-in: this is the only command that keeps running, and nothing starts it
    automatically. Stop it with Ctrl-C or SIGTERM; the slice in flight finishes
    and commits first.
    """
    from hontology.ingest.scheduler import Watcher

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    )
    watcher = Watcher(grace_seconds=grace_seconds, max_slices=max_slices)
    watcher.install_signal_handlers()
    watcher.run()
