"""Ontology portability: JSON and OWL export and import."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from hontology.apps.api.routers.ontology.lookups import service_error
from hontology.apps.api.schemas.ontology import OntologyImport, OntologyOut, OwlImport
from hontology.db.session import get_db
from hontology.ontology import portable, service

router = APIRouter(prefix="/ontologies", tags=["ontology"])


@router.get("/{ontology_id}/export")
def export_ontology(ontology_id: int, db: Session = Depends(get_db)):
    try:
        return portable.export_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise service_error(exc) from exc


@router.post("/import", response_model=OntologyOut)
def import_ontology(payload: OntologyImport, db: Session = Depends(get_db)):
    """Create or merge. Re-importing an edited export updates in place."""
    try:
        data = payload.model_dump()
        if data.get("relations") is None:
            # Absent, not empty: leave the ontology's relations as they are.
            data.pop("relations", None)
        return portable.import_ontology(db, data)
    except (service.Conflict, service.NotFound) as exc:
        raise service_error(exc) from exc


@router.get("/{ontology_id}/export.owl", response_class=PlainTextResponse)
def export_owl(ontology_id: int, db: Session = Depends(get_db)):
    """The ontology as OWL (Turtle), for Protégé or any OWL tool."""
    from hontology.ontology import owl

    try:
        text = owl.export_turtle(db, ontology_id)
    except service.NotFound as exc:
        raise service_error(exc) from exc
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
        raise service_error(exc) from exc
    except (ParserError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"not a readable Turtle file: {exc}"
        ) from exc
