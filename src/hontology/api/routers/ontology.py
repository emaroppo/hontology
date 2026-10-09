"""Ontology endpoints.

The UI reaches every one of these over HTTP and never touches the database, which
is what keeps the Streamlit layer replaceable without moving any logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.api.schemas.ontology import (
    ConceptIn,
    ConceptOut,
    ConceptPatch,
    OntologyImport,
    OntologyIn,
    OntologyOut,
    OwlImport,
    SnapshotOut,
)
from hontology.db.models import OntologySnapshot
from hontology.db.models.ontology import PRECURSOR_OF
from hontology.db.session import get_db
from hontology.ontology import hierarchy, service, snapshots

router = APIRouter(prefix="/ontologies", tags=["ontology"])


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, service.NotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


def _ontology_or_404(db: Session, ontology_id: int):
    try:
        return service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


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
    return _ontology_or_404(db, ontology_id)


@router.delete("/{ontology_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        service.delete_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc


# --- Concepts ---------------------------------------------------------------


@router.get("/{ontology_id}/concepts", response_model=list[ConceptOut])
def list_concepts(ontology_id: int, db: Session = Depends(get_db)):
    _ontology_or_404(db, ontology_id)
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


# --- Structure --------------------------------------------------------------


@router.get("/{ontology_id}/hierarchy")
def get_hierarchy(ontology_id: int, db: Session = Depends(get_db)):
    """Every class with its place in the hierarchy, for display.

    Read-only on purpose: structure is authored in an OWL editor and arrives by
    import, so there is nothing here to edit it with.
    """
    _ontology_or_404(db, ontology_id)
    concepts = service.list_concepts(db, ontology_id)
    names = {c.id: c.name for c in concepts}
    categories = {c.id: c.name for c in service.list_categories(db, ontology_id)}
    parent_map = hierarchy.parents(db, ontology_id)
    child_map = hierarchy.children(db, ontology_id)
    precursors: dict[int, list[int]] = {}
    for subject, obj in hierarchy.edges(db, ontology_id, PRECURSOR_OF):
        precursors.setdefault(subject, []).append(obj)
    groups: dict[int, list[str]] = {}
    for group in service.list_groups(db, ontology_id):
        for member in group.members:
            groups.setdefault(member.concept_id, []).append(group.name)

    def by_name(ids) -> list[str]:
        return sorted(names[i] for i in ids)

    return {
        "structured": bool(parent_map),
        "classes": [
            {
                "id": c.id,
                "name": c.name,
                "definition": c.definition,
                "inclusion_criteria": c.inclusion_criteria,
                "exclusion_criteria": c.exclusion_criteria,
                "category": categories.get(c.category_id) if c.category_id else None,
                "weight": c.weight,
                "leaf": c.id not in child_map,
                "parents": by_name(parent_map.get(c.id, ())),
                "children": by_name(child_map.get(c.id, ())),
                "precursor_of": by_name(precursors.get(c.id, ())),
                "groups": sorted(groups.get(c.id, ())),
            }
            for c in concepts
        ],
    }


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
        data = payload.model_dump()
        if data.get("relations") is None:
            # Absent, not empty: leave the ontology's relations as they are.
            data.pop("relations", None)
        return service.import_ontology(db, data)
    except (service.Conflict, service.NotFound) as exc:
        raise _handle(exc) from exc


@router.get("/{ontology_id}/export.owl", response_class=PlainTextResponse)
def export_owl(ontology_id: int, db: Session = Depends(get_db)):
    """The ontology as OWL (Turtle), for Protégé or any OWL tool."""
    from hontology.ontology import owl

    try:
        text = owl.export_turtle(db, ontology_id)
    except service.NotFound as exc:
        raise _handle(exc) from exc
    return PlainTextResponse(text, media_type="text/turtle")


@router.post("/import-owl", response_model=OntologyOut)
def import_owl(payload: OwlImport, db: Session = Depends(get_db)):
    """Create or merge from OWL (Turtle), relations included.

    Rewording existing classes is refused unless asked for, as on the command
    line: this is the path a hierarchy arrives by, and labels depend on wording.
    """
    from rdflib.exceptions import ParserError

    from hontology.ontology import owl

    try:
        return owl.import_turtle(
            db, payload.turtle, allow_text_change=payload.allow_text_change
        )
    except (service.Conflict, service.NotFound) as exc:
        raise _handle(exc) from exc
    except (ParserError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"not a readable Turtle file: {exc}"
        ) from exc


# --- Versioning -------------------------------------------------------------


@router.post("/{ontology_id}/snapshot", response_model=SnapshotOut)
def resolve_snapshot(ontology_id: int, db: Session = Depends(get_db)):
    """Resolve the live concept set to a version, minting one only if it changed.

    Safe to call repeatedly: an unchanged ontology keeps returning the same
    version with ``created=false``.
    """
    _ontology_or_404(db, ontology_id)
    ref = snapshots.resolve_current(db, ontology_id)
    return SnapshotOut(
        version=ref.version,
        content_hash=ref.content_hash,
        n_concepts=ref.n_concepts,
        created=ref.created,
    )


@router.get("/{ontology_id}/versions")
def list_versions(ontology_id: int, db: Session = Depends(get_db)):
    """Minted versions, oldest first, and which one the live wording is.

    ``current`` is None when the wording has changed since the last version;
    the next run or label mints the new one, so nothing is minted here.
    """
    _ontology_or_404(db, ontology_id)
    rows = db.scalars(
        select(OntologySnapshot)
        .where(OntologySnapshot.ontology_id == ontology_id)
        .order_by(OntologySnapshot.id)
    )
    return {
        "current": snapshots.current_version(db, ontology_id),
        "versions": [
            {"version": r.version, "n_concepts": r.n_concepts, "created_at": r.created_at}
            for r in rows
        ],
    }


@router.get("/{ontology_id}/lint")
def lint_ontology(ontology_id: int, db: Session = Depends(get_db)):
    """Health checks: strength drift, near-duplicates, thin definitions.

    Advisory only — an ontology is the user's to author, and a lint that blocks
    is a lint that gets ignored.
    """
    from hontology.ontology import lint as lint_module

    _ontology_or_404(db, ontology_id)
    return lint_module.lint(db, ontology_id)
