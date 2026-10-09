"""Ontologies: list, create, fetch, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from hontology.apps.api.routers.ontology.lookups import ontology_or_404, service_error
from hontology.apps.api.schemas.ontology import OntologyIn, OntologyOut
from hontology.db.session import get_db
from hontology.ontology import service

router = APIRouter(prefix="/ontologies", tags=["ontology"])


@router.get("", response_model=list[OntologyOut])
def list_ontologies(db: Session = Depends(get_db)):
    return service.list_ontologies(db)


@router.post("", response_model=OntologyOut, status_code=status.HTTP_201_CREATED)
def create_ontology(payload: OntologyIn, db: Session = Depends(get_db)):
    try:
        return service.create_ontology(
            db, slug=payload.slug, name=payload.name, description=payload.description
        )
    except (service.Conflict, service.NotFound) as exc:
        raise service_error(exc) from exc


@router.get("/{ontology_id}", response_model=OntologyOut)
def get_ontology(ontology_id: int, db: Session = Depends(get_db)):
    return ontology_or_404(db, ontology_id)


@router.delete("/{ontology_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        service.delete_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise service_error(exc) from exc
