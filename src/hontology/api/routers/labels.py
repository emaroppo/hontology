"""Ground-truth bank endpoints: the labelling queue, labels, observations."""

from __future__ import annotations

from datetime import date as _date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Concept, Document
from hontology.db.session import get_db
from hontology.evalkit import document_labels, label_io
from hontology.evalkit import labels as label_service
from hontology.evalkit import queue as queue_service
from hontology.ontology import hierarchy

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


class PendingOut(BaseModel):
    id: int
    document_id: int
    concept_id: int
    document_url: str
    document_title: str | None = None
    concept_name: str
    proposed_matched: bool
    proposed_by: str | None = None
    note: str | None = None
    ontology_version: str | None = None
    stale: bool = False


def _pending_rows(db: Session, rows, stale_ids: set[int]) -> list[PendingOut]:
    from hontology.db.models import Concept, Document

    out: list[PendingOut] = []
    for label in rows:
        document = db.get(Document, label.document_id)
        concept = db.get(Concept, label.concept_id)
        if document is None or concept is None:
            continue
        out.append(
            PendingOut(
                id=label.id,
                document_id=label.document_id,
                concept_id=label.concept_id,
                document_url=document.url,
                document_title=document.title,
                concept_name=concept.name,
                proposed_matched=label.matched,
                proposed_by=label.proposed_by,
                note=label.note,
                ontology_version=label.ontology_version,
                stale=label.id in stale_ids,
            )
        )
    return out


@router.get("/pending", response_model=list[PendingOut])
def pending_adjudication(ontology_id: int, limit: int = 50, db: Session = Depends(get_db)):
    """Machine proposals that do not count until a human confirms or flips them."""
    rows = label_service.pending_adjudication(db, ontology_id, limit=limit)
    return _pending_rows(db, rows, set())


@router.get("/stale/detail", response_model=list[PendingOut])
def stale_detail(ontology_id: int, limit: int = 50, db: Session = Depends(get_db)):
    """Stale labels, ready to be re-adjudicated against the current wording."""
    rows = label_service.stale_labels(db, ontology_id, limit=limit)
    return _pending_rows(db, rows, {row.id for row in rows})


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


# ---------------------------------------------------------------------------
# Portability
#
# The bank is hours of human attention, so it has to travel: reviewable in a
# diff, safe to re-import, and keyed by names rather than local database ids.
# ---------------------------------------------------------------------------


class ImportIn(BaseModel):
    ontology_id: int
    csv: str
    # Off by default: importing must never silently destroy adjudicated work.
    overwrite: bool = False
    create_missing_documents: bool = True


@router.get("/export", response_class=PlainTextResponse)
def export_labels(ontology_id: int, db: Session = Depends(get_db)):
    """All pair labels as CSV. Stale ones are included and flagged, not dropped."""
    return PlainTextResponse(
        label_io.export_labels(db, ontology_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="labels-{ontology_id}.csv"'},
    )


@router.get("/observations/export", response_class=PlainTextResponse)
def export_observations(ontology_id: int, db: Session = Depends(get_db)):
    """Known occurrences as CSV."""
    return PlainTextResponse(
        label_io.export_observations(db, ontology_id),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="observations-{ontology_id}.csv"'
        },
    )


@router.post("/import")
def import_labels(payload: ImportIn, db: Session = Depends(get_db)):
    """Load labels from CSV, matching on document URL and concept name."""
    return label_io.import_labels(
        db,
        payload.ontology_id,
        payload.csv,
        overwrite=payload.overwrite,
        create_missing_documents=payload.create_missing_documents,
    )


@router.post("/observations/import")
def import_observations(payload: ImportIn, db: Session = Depends(get_db)):
    """Load known occurrences from CSV."""
    return label_io.import_observations(db, payload.ontology_id, payload.csv)


# --- whole-document labelling ------------------------------------------------
#
# A labelled sample is labelled a document at a time: every leaf answered for
# the document at once, the ones that apply positive and the rest negative.
# Nothing here returns a run's verdicts, so a sample can be labelled blind.

# Long enough for any article; the judge reads a shorter prefix.
DOCUMENT_BODY_LIMIT = 200_000


class DocumentStatusIn(BaseModel):
    ontology_id: int
    document_ids: list[int]


class DocumentLabelIn(BaseModel):
    ontology_id: int
    concept_ids: list[int]
    note: str | None = None


@router.get("/leaves")
def leaves(ontology_id: int, db: Session = Depends(get_db)):
    """The leaves a document is labelled against, each with its top-level families."""
    leaf_ids = hierarchy.leaves(db, ontology_id)
    parent_map = hierarchy.parents(db, ontology_id)
    concepts = {
        c.id: c for c in db.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    out = []
    for concept_id in leaf_ids:
        concept = concepts[concept_id]
        above = hierarchy.ancestors(parent_map, concept_id)
        families = sorted(concepts[a].name for a in above if a not in parent_map)
        out.append(
            {
                "id": concept.id,
                "name": concept.name,
                "definition": concept.definition,
                "inclusion_criteria": concept.inclusion_criteria,
                "exclusion_criteria": concept.exclusion_criteria,
                "families": families or [concept.name],
            }
        )
    return sorted(out, key=lambda leaf: (leaf["families"][0], leaf["name"]))


@router.post("/documents/status")
def documents_status(payload: DocumentStatusIn, db: Session = Depends(get_db)):
    """Each document's labelling state, in the order asked."""
    status_by_id = document_labels.document_label_status(
        db, payload.ontology_id, payload.document_ids
    )
    documents = {
        d.id: d
        for d in db.scalars(select(Document).where(among(Document.id, payload.document_ids)))
    }
    return [
        {
            "document_id": d,
            "url": documents[d].url if d in documents else None,
            "title": documents[d].title if d in documents else None,
            **status_by_id[d],
        }
        for d in payload.document_ids
    ]


@router.get("/documents/{document_id}")
def document_for_labelling(document_id: int, ontology_id: int, db: Session = Depends(get_db)):
    """One document's text and its current labels."""
    from hontology.retrieve.candidates import document_body

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"document {document_id} not found")
    state = document_labels.document_label_status(db, ontology_id, [document_id])[document_id]
    return {
        "document_id": document.id,
        "url": document.url,
        "title": document.title,
        "body": document_body(document, DOCUMENT_BODY_LIMIT),
        **state,
    }


@router.put("/documents/{document_id}")
def label_document(document_id: int, payload: DocumentLabelIn, db: Session = Depends(get_db)):
    """Answer every leaf for one document: the listed ones apply, the rest do not."""
    if db.get(Document, document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"document {document_id} not found")
    try:
        report = document_labels.label_document(
            db,
            payload.ontology_id,
            document_id,
            set(payload.concept_ids),
            note=(payload.note or "").strip() or None,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return report


@router.get("/truths")
def truths(ontology_id: int, db: Session = Depends(get_db)):
    """What a score can be computed against: the human label bank, and each
    machine annotation set, with how much each covers."""
    from hontology.evalkit import annotations

    return {
        "human": annotations.human_summary(db, ontology_id),
        "machine": annotations.list_sets(db, ontology_id),
    }
