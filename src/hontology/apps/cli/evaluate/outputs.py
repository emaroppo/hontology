"""`hontology eval` on what a run produced: its funnel and its detections."""

from __future__ import annotations

from pathlib import Path

import typer

from hontology.apps.cli.common import emit
from hontology.db.session import session_scope


def eval_funnel(run_id: int) -> None:
    """Where the volume went, stage by stage. Needs no labels."""
    from hontology.evaluation.outputs import funnel

    with session_scope() as session:
        typer.echo(funnel.format_funnel(funnel.funnel(session, run_id)))


def eval_detections(
    run_id: int,
    out: Path | None = typer.Option(None, help="Write CSV to a file instead of stdout."),
    events: bool = typer.Option(False, help="Aggregate to one row per (concept, locus, date)."),
    min_confidence: float | None = typer.Option(None, help="Drop weaker detections."),
    verified_only: bool = typer.Option(False, help="Only detections a human has confirmed."),
) -> None:
    """Export what the pipeline found. Every row carries its verification status."""
    from hontology.evaluation.outputs import detections as detections_module

    exporter = (
        detections_module.export_events if events else detections_module.export_detections
    )
    with session_scope() as session:
        text_out = exporter(
            session, run_id, min_confidence=min_confidence, verified_only=verified_only
        )
        stats = detections_module.summary(session, run_id)
    emit(text_out, out, count_rows=True)

    typer.echo(
        f"{stats['detections']} detection(s) over {stats['events']} event(s); "
        f"{stats['by_verification']}"
    )
    if stats["by_verification"].get("unverified"):
        typer.secho(
            "unverified detections are model claims, not facts — adjudicate before "
            "treating them as findings",
            fg=typer.colors.YELLOW,
        )
