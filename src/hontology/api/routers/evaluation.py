"""Evaluation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
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
            "stage": run.stage,
            "progress_done": run.progress_done,
            "progress_total": run.progress_total,
            "error": run.error,
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


@router.get("/runs/{run_id}/sample")
def run_on_sample(run_id: int, manifest: str, db: Session = Depends(get_db)):
    """One run's end-to-end scores on a labelled sample, with the pairs it got wrong.

    *manifest* is the sample manifest's path on the API's machine.
    """
    return _sample(db, run_id, manifest, None)


class SampleIn(BaseModel):
    manifest_path: str
    # The truth: None for human labels, else a machine annotation set's name.
    annotator: str | None = None


@router.post("/runs/{run_id}/sample")
def run_on_sample_with(run_id: int, payload: SampleIn, db: Session = Depends(get_db)):
    """As the GET, scored against a machine annotation set if one is named."""
    return _sample(db, run_id, payload.manifest_path, payload.annotator)


def _manifest(path: str) -> dict:
    import json
    from pathlib import Path

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"cannot read manifest: {exc}"
        ) from exc


def _labels(db: Session, ontology_id: int, annotator: str | None) -> dict | None:
    """None for human labels, which the sample scoring reads from the bank itself."""
    from hontology.retrieve import tuning

    if annotator is None:
        return None
    try:
        return tuning.truth(db, ontology_id, annotator)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _sample(db: Session, run_id: int, manifest_path: str, annotator: str | None) -> dict:
    from hontology.db.models import Concept, Document
    from hontology.evalkit import arms

    record = _manifest(manifest_path)
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id} does not exist")
    labels = _labels(db, run.ontology_id, annotator)
    result = arms.run_on_sample(db, run_id, record, labels=labels)
    for error in result.get("errors", []):
        concept = db.get(Concept, error["concept_id"])
        document = db.get(Document, error["document_id"])
        error["concept"] = concept.name if concept else None
        error["document_url"] = document.url if document else None
        error["document_title"] = document.title if document else None
    return result


@router.get("/runs/{run_id}/breakdown")
def run_breakdown(
    run_id: int,
    dimension: str = "concept",
    include_machine: bool = False,
    db: Session = Depends(get_db),
):
    """Metrics sliced by concept, family, category or locus, each with its own interval."""
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


@router.get("/runs/{run_id}/funnel")
def run_funnel(run_id: int, db: Session = Depends(get_db)):
    """Stage-by-stage attrition. Works with no ground truth at all."""
    from hontology.evalkit import funnel

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
    from hontology.evalkit import detections as detections_module

    try:
        if events:
            rows = [
                e.as_row()
                for e in detections_module.events(
                    db, run_id, min_confidence=min_confidence, verified_only=verified_only
                )
            ]
        else:
            rows = [
                d.as_row()
                for d in detections_module.detections(
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
    from hontology.evalkit import detections as detections_module

    exporter = (
        detections_module.export_events if events else detections_module.export_detections
    )
    try:
        body = exporter(db, run_id, min_confidence=min_confidence, verified_only=verified_only)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    kind = "events" if events else "detections"
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{kind}-run{run_id}.csv"'},
    )


# --- Live leaderboard, comparison and single-run summary ---------------------


class LeaderboardIn(BaseModel):
    ontology_id: int
    manifest_path: str
    # Also score runs that worked on only part of the sample (slower).
    include_partial: bool = False
    annotator: str | None = None


@router.post("/leaderboard/live")
def live_leaderboard(payload: LeaderboardIn, db: Session = Depends(get_db)):
    """Every judged run of an ontology scored end to end on the labelled sample."""
    from hontology.evalkit import leaderboard

    return leaderboard.rows(
        db,
        payload.ontology_id,
        _manifest(payload.manifest_path),
        _labels(db, payload.ontology_id, payload.annotator),
        include_partial=payload.include_partial,
    )


class CalendarIn(BaseModel):
    # None: the calendar the run worked through.
    calendar_path: str | None = None


@router.post("/runs/{run_id}/calendar")
def run_calendar(run_id: int, payload: CalendarIn, db: Session = Depends(get_db)):
    """A run's calendar scores. Slow: every window is walked."""
    from hontology.evalkit import leaderboard

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
    from hontology.evalkit import arms, arms_report

    baseline = db.get(Run, payload.baseline)
    if baseline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {payload.baseline} does not exist")
    report = arms.compare_arms(
        db,
        payload.baseline,
        payload.arms,
        [],
        _manifest(payload.manifest_path),
        labels=_labels(db, baseline.ontology_id, payload.annotator),
    )
    return {"report": report, "markdown": arms_report.render_markdown(report)}
