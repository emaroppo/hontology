"""Command line entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from sqlalchemy import text

from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.ingest import service
from hontology.judge.providers.base import ProviderError
from hontology.judge.providers.ollama import OllamaChatProvider
from hontology.ontology import service as ontology_service
from hontology.ontology import snapshots

app = typer.Typer(help="hontology — ontology-driven event detection and evaluation.")
ontology_app = typer.Typer(help="Manage ontologies.")
ingest_app = typer.Typer(help="Pull slices from the news feed.")
app.add_typer(ontology_app, name="ontology")
app.add_typer(ingest_app, name="ingest")


@app.command()
def doctor() -> None:
    """Check that the things this project depends on are reachable."""
    settings = get_settings()
    ok = True

    typer.echo(f"data dir       {settings.data_dir.resolve()}")

    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        typer.echo(f"database       ok  ({settings.psycopg_url()})")
    except Exception as exc:  # noqa: BLE001 - a diagnostic should never crash
        ok = False
        typer.secho(f"database       FAIL  {exc}", fg=typer.colors.RED)

    try:
        typer.echo(f"llm provider   {OllamaChatProvider(settings.ollama_host).health()}")
    except ProviderError as exc:
        # Not fatal: the ontology layer works fine without a model.
        typer.secho(f"llm provider   unavailable  ({exc})", fg=typer.colors.YELLOW)

    raise typer.Exit(0 if ok else 1)


@ontology_app.command("list")
def ontology_list() -> None:
    """List ontologies and their concept counts."""
    with session_scope() as session:
        rows = ontology_service.list_ontologies(session)
        if not rows:
            typer.echo("(none — this install is empty by design)")
            return
        for ontology in rows:
            n = len(ontology_service.list_concepts(session, ontology.id))
            typer.echo(f"{ontology.id:>4}  {ontology.slug:<24} {ontology.name}  [{n} concepts]")


@ontology_app.command("import")
def ontology_import(path: Path) -> None:
    """Create or merge an ontology from a JSON export."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    with session_scope() as session:
        ontology = ontology_service.import_ontology(session, payload)
        session.flush()
        typer.echo(f"imported {ontology.slug!r} (id {ontology.id})")


@ontology_app.command("export")
def ontology_export(ontology_id: int, out: Path | None = None) -> None:
    """Write an ontology to a portable JSON file, or stdout."""
    with session_scope() as session:
        payload = ontology_service.export_ontology(session, ontology_id)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")


@ontology_app.command("version")
def ontology_version(ontology_id: int) -> None:
    """Resolve the current version, minting one only if the wording changed."""
    with session_scope() as session:
        ref = snapshots.resolve_current(session, ontology_id)
    status = "minted" if ref.created else "unchanged"
    typer.echo(f"{ref.version}  ({status}, {ref.n_concepts} concepts)")


# ---------------------------------------------------------------------------
# Ingest
#
# One-shot commands are the baseline. `watch` is the only continuous one, and it
# has to be started deliberately — nothing else in the system spawns it.
# ---------------------------------------------------------------------------


@ingest_app.command("status")
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


@ingest_app.command("once")
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


@ingest_app.command("backfill")
def ingest_backfill(start: str, end: str) -> None:
    """Ingest an explicit window. Does not move the watermark."""
    with session_scope() as session:
        result = service.backfill(session, start, end)
    typer.echo(f"processed {result['processed']} slice(s) from {start} to {end}")


@ingest_app.command("watch")
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


if __name__ == "__main__":
    app()
