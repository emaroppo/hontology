"""Filter evaluation: a set of links scored against labels, calendar and cost.

A version is a link snapshot (``f1``, ...) or, when none is given, the live
links. See `evalkit.filter_eval` for what each count means.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.api.routers._common import truth_or_404
from hontology.db.models import Concept, LinkSnapshot, Run
from hontology.db.session import get_db
from hontology.evalkit import calendar, filter_eval, versions

router = APIRouter(prefix="/filtering", tags=["filtering"])


class EvaluateIn(BaseModel):
    ontology_id: int
    # None: the live links.
    version: str | None = None
    annotator: str | None = None
    calendar_path: str | None = None
    before: int = 1
    after: int = 2


def _links(db: Session, payload: EvaluateIn) -> list[list]:
    try:
        return filter_eval.link_set(db, payload.ontology_id, payload.version)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _names(db: Session, ontology_id: int) -> dict[int, str]:
    return {
        cid: name
        for cid, name in db.execute(
            select(Concept.id, Concept.name).where(Concept.ontology_id == ontology_id)
        )
    }


@router.get("/versions")
def list_versions(ontology_id: int, db: Session = Depends(get_db)):
    """Link snapshots, newest first, with the runs that fetched with each, and
    which one (if any) the live links currently are."""
    live = versions.current_links(db, ontology_id)
    used: dict[str, list[int]] = {}
    for run in db.scalars(select(Run).where(Run.ontology_id == ontology_id)):
        for version in versions.filter_versions(run):
            used.setdefault(version, []).append(run.id)
    snapshots = list(
        db.scalars(
            select(LinkSnapshot)
            .where(LinkSnapshot.ontology_id == ontology_id)
            .order_by(LinkSnapshot.id.desc())
        )
    )
    return {
        "live": {
            "n_links": len(live),
            "is": next((s.version for s in snapshots if json.loads(s.payload) == live), None),
        },
        "snapshots": [
            {
                "version": s.version,
                "n_links": s.n_links,
                "created_at": s.created_at,
                "runs": sorted(used.get(s.version, [])),
            }
            for s in snapshots
        ],
    }


@router.post("/evaluate/labels")
def evaluate_labels(payload: EvaluateIn, db: Session = Depends(get_db)):
    truth = truth_or_404(db, payload.ontology_id, payload.annotator)
    return filter_eval.labelled_report(
        db, _links(db, payload), truth, _names(db, payload.ontology_id)
    )


@router.post("/evaluate/calendar")
def evaluate_calendar(payload: EvaluateIn, db: Session = Depends(get_db)):
    """Slow: walks every calendar window. The calendar path is read on this host."""
    if not payload.calendar_path:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "no calendar given")
    try:
        entries = calendar.load(Path(payload.calendar_path))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"cannot read the calendar: {exc}"
        ) from exc
    return filter_eval.calendar_report(
        db,
        _links(db, payload),
        entries,
        _names(db, payload.ontology_id),
        before=payload.before,
        after=payload.after,
    )


@router.post("/evaluate/cost")
def evaluate_cost(payload: EvaluateIn, db: Session = Depends(get_db)):
    """Slow: walks every feed record."""
    return filter_eval.corpus_cost(db, _links(db, payload))


@router.post("/versions/snapshot")
def snapshot(ontology_id: int, db: Session = Depends(get_db)):
    """Keep the live links as a version, minting one only if they changed."""
    found = versions.resolve_links(db, ontology_id)
    if found is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "this ontology has no links")
    return {"version": found.version, "n_links": found.n_links}
