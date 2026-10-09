"""Live leaderboard, arms comparison and a run's calendar scores."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hontology.apps.api.routers._common import run_or_404
from hontology.apps.api.routers.evaluation.sample import read_manifest, sample_labels
from hontology.db.session import get_db

router = APIRouter(prefix="/eval", tags=["evaluation"])


class LeaderboardIn(BaseModel):
    ontology_id: int
    manifest_path: str
    # Also score runs that worked on only part of the sample (slower).
    include_partial: bool = False
    annotator: str | None = None


@router.post("/leaderboard/live")
def live_leaderboard(payload: LeaderboardIn, db: Session = Depends(get_db)):
    """Every judged run of an ontology scored end to end on the labelled sample."""
    from hontology.evaluation.comparison import leaderboard

    return leaderboard.rows(
        db,
        payload.ontology_id,
        read_manifest(payload.manifest_path),
        sample_labels(db, payload.ontology_id, payload.annotator),
        include_partial=payload.include_partial,
    )


class CalendarIn(BaseModel):
    # None: the calendar the run worked through.
    calendar_path: str | None = None


@router.post("/runs/{run_id}/calendar")
def run_calendar(run_id: int, payload: CalendarIn, db: Session = Depends(get_db)):
    """A run's calendar scores. Slow: every window is walked."""
    from hontology.evaluation.comparison import leaderboard

    try:
        return leaderboard.calendar_recall(db, run_id, payload.calendar_path)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


class ArmsIn(BaseModel):
    baseline: int
    arms: list[int]
    manifest_path: str
    annotator: str | None = None


@router.post("/arms")
def compare_arms(payload: ArmsIn, db: Session = Depends(get_db)):
    """Arms against a baseline on the labelled sample, paired, as `eval arms`
    reports them, with the same Markdown tables. Calendar scores are left to
    each run's own view, since they take minutes."""
    from hontology.evaluation.comparison import arms, arms_report

    baseline = run_or_404(db, payload.baseline)
    report = arms.compare_arms(
        db,
        payload.baseline,
        payload.arms,
        [],
        read_manifest(payload.manifest_path),
        labels=sample_labels(db, baseline.ontology_id, payload.annotator),
    )
    return {"report": report, "markdown": arms_report.render_markdown(report)}
