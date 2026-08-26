"""Ground-truth bank endpoints: the labelling queue, labels, observations."""

from __future__ import annotations

from datetime import date as _date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hontology.db.session import get_db
from hontology.evalkit import labels as label_service
from hontology.evalkit import queue as queue_service

router = APIRouter(prefix="/labels", tags=["labels"])


class LabelIn(BaseModel):
    document_id: int
    concept_id: int
    matched: bool
    source: str = label_service.HUMAN
    locus_id: int | None = None
    occurred_on: _date | None = None
    note: str | None = None
    proposed_by: str | None = None


class AdjudicateIn(BaseModel):
    matched: bool
    note: str | None = None


class LabelOut(BaseModel):
    id: int
    document_id: int
    concept_id: int
    matched: bool
    source: str
    ontology_version: str | None = None


class ObservationIn(BaseModel):
    concept_id: int
    locus_id: int
    occurred_on: _date
    description: str | None = None
    source_note: str | None = None


@router.get("/queue")
def labelling_queue(
    ontology_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    per_concept_cap: int | None = None,
    include_unjudged: bool = True,
    db: Session = Depends(get_db),
):
    """Unlabelled pairs, ranked by how much a label would teach.

    Disagreement between runs first, then genuine model uncertainty, then
    coverage. See `evalkit.queue` for why confidence is not always trusted.
    """
    items = queue_service.build_queue(
        db,
        ontology_id,
        limit=limit,
        per_concept_cap=per_concept_cap,
        include_unjudged=include_unjudged,
    )
    return [item.as_dict() for item in items]


@router.get("/stats")
def label_stats(ontology_id: int, db: Session = Depends(get_db)):
    """Counts by source and polarity, plus how many labels have gone stale."""
    return label_service.stats(db, ontology_id)


@router.post("", response_model=LabelOut, status_code=status.HTTP_201_CREATED)
def create_label(payload: LabelIn, db: Session = Depends(get_db)):
    try:
        label = label_service.upsert_label(db, **payload.model_dump())
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return LabelOut(
        id=label.id,
        document_id=label.document_id,
        concept_id=label.concept_id,
        matched=label.matched,
        source=label.source,
        ontology_version=label.ontology_version,
    )


@router.post("/{label_id}/adjudicate", response_model=LabelOut)
def adjudicate_label(label_id: int, payload: AdjudicateIn, db: Session = Depends(get_db)):
    """Confirm or flip a machine proposal, promoting it to ground truth."""
    try:
        label = label_service.adjudicate(
            db, label_id, matched=payload.matched, note=payload.note
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return LabelOut(
        id=label.id,
        document_id=label.document_id,
        concept_id=label.concept_id,
        matched=label.matched,
        source=label.source,
        ontology_version=label.ontology_version,
    )


@router.get("/stale")
def stale_labels(ontology_id: int, db: Session = Depends(get_db)):
    """Labels whose concept has been reworded since they were made."""
    ids = sorted(label_service.stale_label_ids(db, ontology_id))
    return {"count": len(ids), "label_ids": ids}


@router.post("/observations", status_code=status.HTTP_201_CREATED)
def create_observation(payload: ObservationIn, db: Session = Depends(get_db)):
    """Assert a known occurrence. Positive only, and independent of any run."""
    observation = label_service.record_observation(db, **payload.model_dump())
    return {
        "id": observation.id,
        "concept_id": observation.concept_id,
        "locus_id": observation.locus_id,
        "occurred_on": observation.occurred_on,
    }


@router.get("/observations/{observation_id}/documents")
def observation_documents(observation_id: int, db: Session = Depends(get_db)):
    """Supporting documents, derived from positive labels rather than stored."""
    return [
        {"id": d.id, "url": d.url, "title": d.title}
        for d in label_service.supporting_documents(db, observation_id)
    ]
