"""Ontology endpoints.

The UI reaches every one of these over HTTP and never touches the database, which
is what keeps the Streamlit layer replaceable without moving any logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from hontology.api.schemas.ontology import (
    ConceptIn,
    ConceptOut,
    ConceptPatch,
    OntologyImport,
    OntologyIn,
    OntologyOut,
    SnapshotOut,
)
from hontology.db.session import get_db
from hontology.ontology import service, snapshots

router = APIRouter(prefix="/ontologies", tags=["ontology"])


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, service.NotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


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
        raise _handle(exc) from exc


@router.get("/{ontology_id}", response_model=OntologyOut)
def get_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        return service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


@router.delete("/{ontology_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        service.delete_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


# --- Concepts ---------------------------------------------------------------


@router.get("/{ontology_id}/concepts", response_model=list[ConceptOut])
def list_concepts(ontology_id: int, db: Session = Depends(get_db)):
    try:
        service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc
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
        raise _handle(exc) from exc


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
        raise _handle(exc) from exc


@router.delete("/{ontology_id}/concepts/{concept_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_concept(ontology_id: int, concept_id: int, db: Session = Depends(get_db)):
    try:
        service.delete_concept(db, concept_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


# --- Portability ------------------------------------------------------------


@router.get("/{ontology_id}/export")
def export_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        return service.export_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


@router.post("/import", response_model=OntologyOut)
def import_ontology(payload: OntologyImport, db: Session = Depends(get_db)):
    """Create or merge. Re-importing an edited export updates in place."""
    try:
        return service.import_ontology(db, payload.model_dump())
    except (service.Conflict, service.NotFound) as exc:
        raise _handle(exc) from exc


# --- Versioning -------------------------------------------------------------


@router.post("/{ontology_id}/snapshot", response_model=SnapshotOut)
def resolve_snapshot(ontology_id: int, db: Session = Depends(get_db)):
    """Resolve the live concept set to a version, minting one only if it changed.

    Safe to call repeatedly: an unchanged ontology keeps returning the same
    version with ``created=false``.
    """
    try:
        service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc
    ref = snapshots.resolve_current(db, ontology_id)
    return SnapshotOut(
        version=ref.version,
        content_hash=ref.content_hash,
        n_concepts=ref.n_concepts,
        created=ref.created,
    )


@router.get("/{ontology_id}/lint")
def lint_ontology(ontology_id: int, db: Session = Depends(get_db)):
    """Health checks: strength drift, near-duplicates, thin definitions.

    Advisory only — an ontology is the user's to author, and a lint that blocks
    is a lint that gets ignored.
    """
    from hontology.ontology import lint as lint_module

    try:
        service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc
    return lint_module.lint(db, ontology_id)
