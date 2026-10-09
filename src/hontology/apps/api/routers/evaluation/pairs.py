"""Pair-level evaluation of runs: metrics, paired comparison, the leaderboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.apps.api.routers.runs import RunOut
from hontology.db.models import Run
from hontology.db.session import get_db
from hontology.evaluation.pairs import compare as compare_service
from hontology.evaluation.pairs import evaluate as evaluate_service
from hontology.evaluation.pairs import warehouse

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/runs", response_model=list[RunOut])
def list_runs(db: Session = Depends(get_db)):
    return [RunOut.of(run) for run in db.scalars(select(Run).order_by(Run.id.desc()))]


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
