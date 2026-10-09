"""`hontology labels`: move the ground-truth bank in and out."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import select

from hontology.apps.cli.common import After, Before, emit, read_json, write_csv
from hontology.config import get_settings
from hontology.db.session import session_scope

app = typer.Typer(help="Move the ground-truth bank in and out.")
Observations = typer.Option(False, help="Use observations instead of labels.")


@app.command("export")
def labels_export(
    ontology_id: int,
    out: Path | None = typer.Option(None, help="Write to a file instead of stdout."),
    observations: bool = Observations,
) -> None:
    """Export the bank as CSV, keyed by document URL and concept name."""
    from hontology.evaluation.labels import csv_export as label_io

    exporter = label_io.export_observations if observations else label_io.export_labels
    with session_scope() as session:
        text_out = exporter(session, ontology_id)
    emit(text_out, out, count_rows=True)


@app.command("sample")
def labels_sample(
    run_id: int,
    calendar_path: Path,
    out: Path = typer.Option(..., help="Manifest JSON to write."),
    seed: int = typer.Option(20261006, help="Fixes the order; record it with the sample."),
    allocation: str = typer.Option(
        "equal-per-window",
        help="equal-per-window (each window an equal share, metrics weighted by "
        "window size) or proportional.",
    ),
    before: Before = 1,
    after: After = 2,
) -> None:
    """Draw the labelled sample: every representative in the calendar's windows
    that the run processed, in a frozen order whose every prefix is a stratified
    random sample."""
    import hashlib

    from hontology.evaluation.calendar import events
    from hontology.evaluation.labels import sample

    entries = events.load(calendar_path)
    digest = hashlib.sha256(calendar_path.read_bytes()).hexdigest()
    with session_scope() as session:
        documents, late = sample.frame(session, run_id, entries, before=before, after=after)
    ordered = sample.frozen_order(documents, seed, allocation)
    record = sample.manifest(
        run_id, digest, seed, ordered, late_arrivals=late, allocation=allocation
    )
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    typer.echo(
        f"{record['size']} document(s) in {len(record['strata'])} strata; "
        f"order {record['order_sha256'][:12]}; "
        f"{late} left out, fetched after the run processed their window"
    )


@app.command("sample-sheet")
def labels_sample_sheet(
    ontology_id: int,
    manifest_path: Path,
    out: Path = typer.Option(..., help="Labelling CSV to write."),
    start: int = typer.Option(1, help="First position in the frozen order (1-based)."),
    count: int = typer.Option(120, help="How many documents."),
) -> None:
    """Write the next documents to label, in frozen order, plus the concept list.

    Fill `concepts` with the names that apply, separated by `;`, or `none`.
    Leave it blank for a document not yet read; blanks are skipped on import.
    """
    from hontology.db.models import Concept, Document
    from hontology.evaluation.labels.document_labels import DOCUMENT_LABEL_COLUMNS
    from hontology.ontology import hierarchy

    chunk = read_json(manifest_path)["order"][start - 1 : start - 1 + count]
    cache = get_settings().scrape_cache_dir
    with session_scope() as session:
        documents = {
            d.id: d
            for d in session.scalars(
                select(Document).where(Document.id.in_([r["document_id"] for r in chunk]))
            )
        }
        leaf_ids = hierarchy.leaves(session, ontology_id)
        concepts = [
            c
            for c in session.scalars(
                select(Concept).where(Concept.ontology_id == ontology_id).order_by(Concept.name)
            )
            if c.id in leaf_ids  # people label leaves; internal classes are derived
        ]
        rows = []
        for position, row in enumerate(chunk, start=start):
            document = documents[row["document_id"]]
            path = cache / (document.body_path or "")
            body = (
                path.read_text(encoding="utf-8") if document.body_path and path.exists() else ""
            )
            rows.append(
                {
                    "position": position,
                    "document_url": document.url,
                    "title": document.title or body.split("\n", 1)[0][:160],
                    "excerpt": " ".join(body.split())[:700],
                    "concepts": "",
                    "note": "",
                }
            )
        reference = "\n\n".join(
            f"{c.name}\n  {c.definition or ''}\n  counts when: {c.inclusion_criteria or '-'}"
            f"\n  not when: {c.exclusion_criteria or '-'}"
            for c in concepts
        )
    write_csv(out, DOCUMENT_LABEL_COLUMNS, rows)
    concept_file = out.with_suffix(".concepts.txt")
    concept_file.write_text(reference + "\n", encoding="utf-8")
    typer.echo(f"{len(rows)} document(s) to {out}; concept reference in {concept_file}")


@app.command("import-documents")
def labels_import_documents(ontology_id: int, path: Path) -> None:
    """Read whole-document labels: listed concepts positive, all others negative."""
    from hontology.evaluation.labels.document_labels import import_document_labels

    with session_scope() as session:
        report = import_document_labels(session, ontology_id, path.read_text(encoding="utf-8"))
    typer.echo(
        f"{report['documents']} document(s): {report['positives']} positive, "
        f"{report['negatives']} negative label(s); "
        f"{report['skipped_blank']} blank row(s) skipped"
    )
    for error in report["errors"]:
        typer.secho(f"  {error}", fg=typer.colors.YELLOW)


@app.command("annotations-import")
def labels_annotations_import(
    ontology_id: int,
    path: Path,
    name: str = typer.Option(..., help="The set's name, e.g. claude-blind-001."),
    description: str | None = typer.Option(
        None, help="Who annotated and how, for whoever reads a score against it."
    ),
) -> None:
    """Store a whole-document labels file as a machine annotation set.

    Kept apart from the human label bank: scores against it measure agreement
    with that annotator. Re-importing under the same name extends the set.
    """
    from hontology.evaluation.labels import annotations

    with session_scope() as session:
        try:
            report = annotations.import_set(
                session,
                ontology_id,
                name,
                path.read_text(encoding="utf-8"),
                description=description,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"{report['set']}: {report['articles']} article(s), {report['pairs']} pair(s), "
        f"{report['positives']} positive; {report['skipped_blank']} blank row(s) skipped"
    )


@app.command("import")
def labels_import(
    ontology_id: int,
    path: Path,
    overwrite: bool = typer.Option(
        False,
        help="Replace labels that already exist. Off by default so an "
        "import cannot silently destroy adjudicated work.",
    ),
    observations: bool = Observations,
) -> None:
    """Import a CSV bank, matching on document URL and concept name."""
    from hontology.evaluation.labels import csv_io as label_io

    csv_text = path.read_text(encoding="utf-8")
    with session_scope() as session:
        report = (
            label_io.import_observations(session, ontology_id, csv_text)
            if observations
            else label_io.import_labels(session, ontology_id, csv_text, overwrite=overwrite)
        )

    typer.echo(
        f"created {report['created']}, updated {report['updated']}, "
        f"skipped {report['skipped_existing']} existing"
    )
    if report.get("documents_created"):
        typer.echo(f"created {report['documents_created']} stub document(s)")
    if report["unknown_concepts"]:
        typer.secho(
            f"skipped unknown concept(s): {', '.join(report['unknown_concepts'])}",
            fg=typer.colors.YELLOW,
        )
    if report["bad_row_count"]:
        typer.secho(f"{report['bad_row_count']} unreadable row(s):", fg=typer.colors.YELLOW)
        for line in report["bad_rows"]:
            typer.echo(f"  {line}")
