"""`hontology ontology`: manage ontologies."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hontology.apps.cli.common import emit, read_json
from hontology.db.session import session_scope
from hontology.ontology import service as ontology_service
from hontology.ontology import snapshots

app = typer.Typer(help="Manage ontologies.")


@app.command("list")
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


@app.command("import")
def ontology_import(path: Path) -> None:
    """Create or merge an ontology from a JSON export."""
    with session_scope() as session:
        ontology = ontology_service.import_ontology(session, read_json(path))
        session.flush()
        typer.echo(f"imported {ontology.slug!r} (id {ontology.id})")


@app.command("export")
def ontology_export(ontology_id: int, out: Path | None = None) -> None:
    """Write an ontology to a portable JSON file, or stdout."""
    with session_scope() as session:
        payload = ontology_service.export_ontology(session, ontology_id)
    emit(json.dumps(payload, indent=2, ensure_ascii=False), out)


@app.command("export-owl")
def ontology_export_owl(ontology_id: int, out: Path) -> None:
    """Write the ontology as OWL (Turtle), for Protégé or any OWL tool."""
    from hontology.ontology import owl

    with session_scope() as session:
        emit(owl.export_turtle(session, ontology_id), out)


@app.command("import-owl")
def ontology_import_owl(
    path: Path,
    allow_text_change: bool = typer.Option(
        False, help="Permit rewording existing classes; off so structure cannot alter them."
    ),
) -> None:
    """Create or merge an ontology from OWL (Turtle), relations included."""
    from hontology.ontology import owl

    with session_scope() as session:
        ontology = owl.import_turtle(
            session, path.read_text(encoding="utf-8"), allow_text_change=allow_text_change
        )
        ref = snapshots.resolve_current(session, ontology.id)
    typer.echo(
        f"imported {ontology.slug!r} (id {ontology.id}) as {ref.version}"
        f"{' (new version)' if ref.created else ''}"
    )


@app.command("links-import")
def ontology_links_import(ontology_id: int, path: Path) -> None:
    """Apply hand-curated concept↔code links from a CSV (concept,system,code)."""
    import csv

    from hontology.pipeline.retrieve import similarity

    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    with session_scope() as session:
        result = similarity.import_links(session, ontology_id, rows)
    typer.echo(f"linked {result['links']} code(s) across {result['concepts']} concept(s)")


@app.command("lint")
def ontology_lint_command(ontology_id: int) -> None:
    """Health checks: strength drift, near-duplicates, thin definitions."""
    from hontology.ontology import lint as lint_module

    with session_scope() as session:
        result = lint_module.lint(session, ontology_id)

    typer.echo(
        f"{result['concepts']} concept(s): {result['warnings']} warning(s), "
        f"{result['info']} note(s)"
    )
    for finding in result["findings"]:
        colour = typer.colors.YELLOW if finding["severity"] == "warning" else None
        typer.secho(
            f"  [{finding['severity']}] {finding['check']}: {finding['concept_name']}",
            fg=colour,
        )
        typer.echo(f"      {finding['message']}")


@app.command("version")
def ontology_version(ontology_id: int) -> None:
    """Resolve the current version, minting one only if the wording changed."""
    with session_scope() as session:
        ref = snapshots.resolve_current(session, ontology_id)
    status = "minted" if ref.created else "unchanged"
    typer.echo(f"{ref.version}  ({status}, {ref.n_concepts} concepts)")
