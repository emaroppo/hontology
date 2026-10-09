"""Errors and lookups the ontology routers share."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from hontology.ontology import service


def service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, service.NotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


def ontology_or_404(db: Session, ontology_id: int):
    try:
        return service.get_ontology(db, ontology_id)
    except service.NotFound as exc:
        raise service_error(exc) from exc
