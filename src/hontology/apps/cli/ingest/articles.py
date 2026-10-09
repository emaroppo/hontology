"""`hontology ingest` on articles: fetch their text, group near-duplicates, preview."""

from __future__ import annotations

from pathlib import Path

import typer
from sqlalchemy import select

from hontology.apps.cli.common import After, Before, calendar_documents, log_to_console
from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.pipeline.ingest.articles import scrape

# Documents fetched per committed batch by `ingest scrape`.
SCRAPE_BATCH = 200
CalendarScope = typer.Option(
    None, "--calendar", help="Only documents inside this calendar's windows."
)


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
    log_to_console()
    allowed = _scrape_scope(ontology_id, calendar_path, before, after)
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

    if retry_failed:
        # A retry pass would re-attempt the same failures every batch; one is enough.
        first = scrape_batch([])
        totals = {key: first.get(key, 0) for key in scrape.TOTAL_KEYS}
        if totals["attempted"]:
            _report_scrape(totals)
    else:
        totals = scrape.drain(scrape_batch, budget=budget, on_batch=_report_scrape)
    if not totals["attempted"]:
        typer.echo("nothing to fetch")


def _scrape_scope(
    ontology_id: int | None, calendar_path: Path | None, before: int, after: int
) -> list[int] | None:
    """The documents a scrape may fetch, or None for any; exits if none match."""
    from hontology.pipeline.ingest.articles import filter as ingest_filter

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
    return sorted(document_ids) if document_ids is not None else None


def _report_scrape(totals: dict[str, int]) -> None:
    typer.echo(
        f"attempted {totals['attempted']}: {totals['ok']} ok, {totals['junk']} junk, "
        f"{totals['failed']} failed, {totals['blocked_by_robots']} blocked by robots, "
        f"{totals['deferred']} deferred by crawl delay, "
        f"{totals['retry_later']} left for a later batch after a connection failure"
    )


def ingest_dedup(
    calendar_path: Path | None = CalendarScope,
    before: Before = 1,
    after: After = 2,
    threshold: float = typer.Option(0.8, help="Estimated Jaccard similarity to group at."),
) -> None:
    """Group near-duplicate articles so only one of each is retrieved and judged."""
    from hontology.db.models import Document
    from hontology.pipeline.ingest.articles import dedup

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


def ingest_filter_preview(ontology_id: int) -> None:
    """Show what the pre-scrape code filter would keep, without fetching."""
    from hontology.pipeline.ingest.articles import filter as ingest_filter

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
