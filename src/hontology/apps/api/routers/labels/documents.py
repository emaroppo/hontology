"""Whole-document labelling.

A labelled sample is labelled a document at a time: every leaf answered for
the document at once, the ones that apply positive and the rest negative.
Nothing here returns a run's verdicts, so a sample can be labelled blind.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Concept, Document
from hontology.db.session import get_db
from hontology.evaluation.labels import document_labels
from hontology.evaluation.labels.document_labels import document_label_status
from hontology.ontology import hierarchy

router = APIRouter(prefix="/labels", tags=["labels"])

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
    status_by_id = document_label_status(db, payload.ontology_id, payload.document_ids)
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
    from hontology.pipeline.retrieve.candidates import document_body

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"document {document_id} not found")
    state = document_label_status(db, ontology_id, [document_id])[document_id]
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
    from hontology.evaluation.labels import annotations

    return {
        "human": annotations.human_summary(db, ontology_id),
        "machine": annotations.list_sets(db, ontology_id),
    }
