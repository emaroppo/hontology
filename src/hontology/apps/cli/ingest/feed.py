"""`hontology ingest` on the feed itself: watermark, catch-up, backfill, repair, watch."""

from __future__ import annotations

import logging

import typer

from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.pipeline.ingest.feed import catchup, gdelt, repair


def ingest_status() -> None:
    """Show the watermark, how far behind it is, and where the gaps are."""
    with session_scope() as session:
        info = catchup.status(session)

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


def ingest_once(
    max_slices: int = typer.Option(32, help="Upper bound on slices for this pass."),
) -> None:
    """Catch up to the newest published slice, then exit."""
    with session_scope() as session:
        result = catchup.catch_up(session, max_slices=max_slices)
    typer.echo(
        f"latest={result['latest_published']} watermark={result['watermark']} "
        f"processed={result['processed']} remaining={result['remaining']}"
    )
    for row in result["slices"]:
        typer.echo(f"  {row['slice_key']}  {row['status']:<8} {row.get('rows', '')}")


def ingest_backfill(start: str, end: str) -> None:
    """Ingest an explicit window. Does not move the watermark."""
    with session_scope() as session:
        result = catchup.backfill(session, start, end)
    typer.echo(f"processed {result['processed']} slice(s) from {start} to {end}")


def ingest_themes() -> None:
    """Load the GKG theme vocabulary as a code system, for theme links."""
    from hontology.pipeline.ingest.codes import themes

    with session_scope() as session:
        result = themes.ingest(session)
    typer.echo(f"gkg themes: {result['inserted']} new, {result['total']} total")


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

    from hontology.pipeline.ingest.codes.loci import by_fips

    settings = get_settings()
    with session_scope() as session:
        keys = repair.slices_missing_export_loci(session)
        lookup = by_fips(session)
    keys = keys[:limit] if limit is not None else keys
    typer.echo(f"{len(keys)} slice(s) to repair")

    def fetch(key: str) -> tuple[str, dict[str, int] | None]:
        try:
            payload = gdelt.fetch_slice(settings.gdelt_base_url, key)
        except Exception:  # noqa: BLE001 - a failed slice is reported, not fatal
            return key, None
        return key, repair.export_loci(payload, lookup)

    updated = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (key, loci) in enumerate(pool.map(fetch, keys), start=1):
            if loci is None:
                failed += 1
            else:
                with session_scope() as session:
                    updated += repair.apply_export_loci(session, key, loci)
            if done % 500 == 0 or done == len(keys):
                typer.echo(f"[{done}/{len(keys)}] {updated} event(s) located, {failed} failed")


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
    from hontology.pipeline.ingest.feed.scheduler import Watcher

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    )
    watcher = Watcher(grace_seconds=grace_seconds, max_slices=max_slices)
    watcher.install_signal_handlers()
    watcher.run()
