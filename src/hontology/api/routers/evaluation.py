"""Evaluation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Run
from hontology.db.session import get_db
from hontology.evalkit import compare as compare_service
from hontology.evalkit import evaluate as evaluate_service
from hontology.evalkit import warehouse

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    return [
        {
            "id": run.id,
            "name": run.name,
            "status": run.status,
            "ontology_id": run.ontology_id,
            "ontology_version": run.ontology_version,
            "candidates_key": run.candidates_key,
            "judge_key": run.judge_key,
        }
        for run in db.scalars(select(Run).order_by(Run.id.desc()))
    ]


@router.get("/runs/{run_id}")
def evaluate_run(
    run_id: int,
    include_machine: bool = False,
    include_stale: bool = False,
    db: Session = Depends(get_db),
):
    """Per-stage metrics, each with its interval and its denominator."""
    try:
        evaluation = evaluate_service.evaluate_run(
            db, run_id, include_machine=include_machine, include_stale=include_stale
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return evaluation.as_dict()


@router.get("/compare")
def compare_runs(
    run_a: int,
    run_b: int,
    include_machine: bool = False,
    db: Session = Depends(get_db),
):
    """Paired comparison over the pairs both runs judged."""
    try:
        return compare_service.compare_runs(
            db, run_a, run_b, include_machine=include_machine
        ).as_dict()
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/consistency/{run_id}")
def consistency(run_id: int, against: int | None = None, db: Session = Depends(get_db)):
    result = {"cross_document": compare_service.cross_document_agreement(db, run_id)}
    if against is not None:
        result["determinism"] = compare_service.determinism(db, run_id, against)
    return result


@router.get("/leaderboard")
def leaderboard(limit: int = 50):
    return warehouse.leaderboard(limit=limit)


@router.get("/runs/{run_id}/breakdown")
def run_breakdown(
    run_id: int,
    dimension: str = "concept",
    include_machine: bool = False,
    db: Session = Depends(get_db),
):
    """Metrics sliced by concept, category or locus, each with its own interval."""
    from hontology.evalkit import breakdown

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
    from hontology.evalkit import errors

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
    from hontology.evalkit import filter_report

    return filter_report.report(db, ontology_id)
