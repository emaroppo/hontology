"""An ontology's concepts: list, create, update, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from hontology.apps.api.routers.ontology.lookups import ontology_or_404, service_error
from hontology.apps.api.schemas.ontology import ConceptIn, ConceptOut, ConceptPatch
from hontology.db.session import get_db
from hontology.ontology import service

router = APIRouter(prefix="/ontologies", tags=["ontology"])


@router.get("/{ontology_id}/concepts", response_model=list[ConceptOut])
def list_concepts(ontology_id: int, db: Session = Depends(get_db)):
    ontology_or_404(db, ontology_id)
    return service.list_concepts(db, ontology_id)


@router.post(
    "/{ontology_id}/concepts",
    response_model=ConceptOut,
    status_code=status.HTTP_201_CREATED,
)
def create_concept(ontology_id: int, payload: ConceptIn, db: Session = Depends(get_db)):
    try:
        return service.create_concept(db, ontology_id, **payload.model_dump())
    except (service.Conflict, service.NotFound) as exc:
        raise service_error(exc) from exc


@router.patch("/{ontology_id}/concepts/{concept_id}", response_model=ConceptOut)
def update_concept(
    ontology_id: int,
    concept_id: int,
    payload: ConceptPatch,
    db: Session = Depends(get_db),
):
    try:
        return service.update_concept(db, concept_id, **payload.model_dump(exclude_unset=True))
    except (service.Conflict, service.NotFound) as exc:
        raise service_error(exc) from exc


@router.delete("/{ontology_id}/concepts/{concept_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_concept(ontology_id: int, concept_id: int, db: Session = Depends(get_db)):
    try:
        service.delete_concept(db, concept_id)
    except service.NotFound as exc:
        raise service_error(exc) from exc
