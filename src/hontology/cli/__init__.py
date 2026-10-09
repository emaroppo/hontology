"""Command line entry point."""

from __future__ import annotations

import typer
from sqlalchemy import text

from hontology.cli import evaluate, ingest, labels, ontology, runs
from hontology.cli.runs import run_signals
from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.judge import run as judge_run
from hontology.judge.providers.base import ProviderError

__all__ = ["app", "run_signals"]

app = typer.Typer(help="hontology — ontology-driven event detection and evaluation.")
app.add_typer(ontology.app, name="ontology")
app.add_typer(ingest.app, name="ingest")
app.add_typer(runs.app, name="run")
app.add_typer(evaluate.app, name="eval")
app.add_typer(labels.app, name="labels")


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
        provider = judge_run.get_provider(settings.default_judge_provider)
        typer.echo(f"llm provider   {provider.health()}")
    except ProviderError as exc:
        # Not fatal: the ontology layer works fine without a model.
        typer.secho(f"llm provider   unavailable  ({exc})", fg=typer.colors.YELLOW)

    raise typer.Exit(0 if ok else 1)
