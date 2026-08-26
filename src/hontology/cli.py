"""Command line entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from sqlalchemy import select, text

from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.ingest import scrape, service
from hontology.judge.providers.base import ProviderError
from hontology.judge.providers.ollama import OllamaChatProvider
from hontology.ontology import service as ontology_service
from hontology.ontology import snapshots

app = typer.Typer(help="hontology — ontology-driven event detection and evaluation.")
ontology_app = typer.Typer(help="Manage ontologies.")
ingest_app = typer.Typer(help="Pull slices from the news feed.")
app.add_typer(ontology_app, name="ontology")
run_app = typer.Typer(help="Configure and execute detection runs.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(run_app, name="run")


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


@ingest_app.command("scrape")
def ingest_scrape(
    limit: int | None = typer.Option(None, help="Max documents this run (default: budget)."),
    retry_failed: bool = typer.Option(
        False, help="Also re-attempt URLs already recorded as failures."
    ),
    reader_proxy: bool = typer.Option(
        False, help="Allow the third-party reader service as a last resort."
    ),
) -> None:
    """Fetch article text for documents that do not have it yet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    with session_scope() as session:
        result = scrape.scrape_pending(
            session, limit=limit, retry_failed=retry_failed, use_reader_proxy=reader_proxy
        )

    typer.echo(
        f"attempted {result['attempted']}: {result['ok']} ok, {result['junk']} junk, "
        f"{result['failed']} failed, {result['blocked_by_robots']} blocked by robots"
    )
    if result["methods"]:
        typer.echo(f"extractors   {result['methods']}")
    typer.echo(f"remaining    {result['pending_remaining']}")


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


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _load_run(session, run_id: int):
    """Fetch a run or exit with a clear message rather than a traceback."""
    from hontology.db.models import Run

    run = session.get(Run, run_id)
    if run is None:
        typer.secho(f"no run with id {run_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    return run


@run_app.command("prompts")
def run_prompts() -> None:
    """List the registered prompt templates."""
    from hontology.judge import prompts

    for prompt_id in prompts.available():
        template = prompts.get(prompt_id)
        typer.echo(f"{prompt_id:<20} {template.mode}")


@run_app.command("keys")
def run_keys(config_path: Path, ontology_version: str = "v1") -> None:
    """Show the stage keys a config resolves to, without executing anything.

    Useful for checking what an edit will recompute before paying for it.
    """
    from hontology.evalkit import config as run_config

    normalized = run_config.normalize(json.loads(config_path.read_text(encoding="utf-8")))
    keys = run_config.stage_keys(normalized, ontology_version)
    typer.echo(f"candidates  {keys['candidates']}")
    typer.echo(f"judge       {keys['judge']}")


@run_app.command("list")
def run_list() -> None:
    """List runs with their status and stage keys."""
    from hontology.db.models import Run

    with session_scope() as session:
        runs = list(session.scalars(select(Run).order_by(Run.id)))
        if not runs:
            typer.echo("(no runs yet)")
            return
        for run in runs:
            typer.echo(
                f"{run.id:>4}  {run.status:<10} {run.name:<24} "
                f"{run.ontology_version:<5} cand={run.candidates_key} "
                f"judge={run.judge_key.split('_')[0]}"
            )


@run_app.command("start")
def run_start(
    ontology_id: int,
    config_path: Path,
    name: str | None = typer.Option(None, help="Override the config's name."),
    documents: int = typer.Option(100, help="How many documents to consider."),
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged this pass."),
    skip_judge: bool = typer.Option(False, help="Build candidates only."),
) -> None:
    """Execute a run from a JSON config."""
    from hontology.evalkit import runner

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    with session_scope() as session:
        run = runner.create_run(session, ontology_id=ontology_id, config=payload, name=name)
        run_id = run.id
        typer.echo(f"run {run_id}: candidates={run.candidates_key} judge={run.judge_key}")

    with session_scope() as session:
        result = runner.execute(
            session,
            _load_run(session, run_id),
            document_limit=documents,
            judge_limit=judge_limit,
            skip_judge=skip_judge,
        )

    typer.echo(f"candidates  {result['candidates']}")
    if result.get("reused_from_run"):
        typer.secho(
            f"            retrieval reused from run {result['reused_from_run']}",
            fg=typer.colors.GREEN,
        )
    if result.get("judge"):
        typer.echo(f"judge       {result['judge']}")
        live = result["liveness"]
        typer.secho(
            f"liveness    {live['clean']}/{live['verdicts']} clean, {live['errors']} errors",
            fg=typer.colors.GREEN if live["ok"] else typer.colors.RED,
        )


@run_app.command("resume")
def run_resume(
    run_id: int,
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged this pass."),
) -> None:
    """Continue a run, skipping pairs already judged."""
    from hontology.evalkit import runner

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    with session_scope() as session:
        result = runner.execute(session, _load_run(session, run_id), judge_limit=judge_limit)
    typer.echo(f"judge  {result['judge']}")


if __name__ == "__main__":
    app()
