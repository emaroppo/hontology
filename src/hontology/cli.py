"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import text

from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.judge.providers.base import ProviderError
from hontology.judge.providers.ollama import OllamaChatProvider
from hontology.ontology import service, snapshots

app = typer.Typer(help="hontology — ontology-driven event detection and evaluation.")
ontology_app = typer.Typer(help="Manage ontologies.")
app.add_typer(ontology_app, name="ontology")


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
        rows = service.list_ontologies(session)
        if not rows:
            typer.echo("(none — this install is empty by design)")
            return
        for ontology in rows:
            n = len(service.list_concepts(session, ontology.id))
            typer.echo(f"{ontology.id:>4}  {ontology.slug:<24} {ontology.name}  [{n} concepts]")


@ontology_app.command("import")
def ontology_import(path: Path) -> None:
    """Create or merge an ontology from a JSON export."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    with session_scope() as session:
        ontology = service.import_ontology(session, payload)
        session.flush()
        typer.echo(f"imported {ontology.slug!r} (id {ontology.id})")


@ontology_app.command("export")
def ontology_export(ontology_id: int, out: Path | None = None) -> None:
    """Write an ontology to a portable JSON file, or stdout."""
    with session_scope() as session:
        payload = service.export_ontology(session, ontology_id)
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


if __name__ == "__main__":
    app()
