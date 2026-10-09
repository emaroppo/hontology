"""Per-run diagnostics: slices, errors, the funnel, detections, the filter report."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from hontology.apps.api.routers._common import csv_response
from hontology.db.session import get_db

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/runs/{run_id}/breakdown")
def run_breakdown(
    run_id: int,
    dimension: str = "concept",
    include_machine: bool = False,
    db: Session = Depends(get_db),
):
    """Metrics sliced by concept, family, category or locus, each with its own interval."""
    from hontology.evaluation.pairs import breakdown

    try:
        return breakdown.breakdown(
            db, run_id, dimension=dimension, include_machine=include_machine
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/runs/{run_id}/errors")
def run_errors(
    run_id: int,
    kind: str | None = None,
    include_machine: bool = False,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Misclassified pairs with the model's evidence and reasoning attached."""
    from hontology.evaluation.pairs import errors

    try:
        rows = errors.triage(
            db, run_id, kind=kind, include_machine=include_machine, limit=limit
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {
        "rows": [r.as_dict() for r in rows],
        "summary": errors.summary(db, run_id, include_machine=include_machine),
    }


@router.get("/filter-report")
def filter_report_endpoint(ontology_id: int, db: Session = Depends(get_db)):
    """Per-code cost and benefit of the pre-scrape filter mapping."""
    from hontology.evaluation.stages import filter_report

    return filter_report.report(db, ontology_id)


@router.get("/runs/{run_id}/funnel")
def run_funnel(run_id: int, db: Session = Depends(get_db)):
    """Stage-by-stage attrition. Works with no ground truth at all."""
    from hontology.evaluation.outputs import funnel

    try:
        return funnel.funnel(db, run_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/runs/{run_id}/detections")
def run_detections(
    run_id: int,
    events: bool = False,
    min_confidence: float | None = None,
    verified_only: bool = False,
    db: Session = Depends(get_db),
):
    """What the pipeline found, as JSON."""
    from hontology.evaluation.outputs import detections as detections_module

    found = detections_module.events if events else detections_module.detections
    try:
        rows = [
            row.as_row()
            for row in found(
                db, run_id, min_confidence=min_confidence, verified_only=verified_only
            )
        ]
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {"rows": rows, "summary": detections_module.summary(db, run_id)}


@router.get("/runs/{run_id}/detections.csv", response_class=PlainTextResponse)
def run_detections_csv(
    run_id: int,
    events: bool = False,
    min_confidence: float | None = None,
    verified_only: bool = False,
    db: Session = Depends(get_db),
):
    """The same, as CSV, for a downstream consumer."""
    from hontology.evaluation.outputs import detections as detections_module

    exporter = (
        detections_module.export_events if events else detections_module.export_detections
    )
    try:
        body = exporter(db, run_id, min_confidence=min_confidence, verified_only=verified_only)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    kind = "events" if events else "detections"
    return csv_response(body, f"{kind}-run{run_id}.csv")
