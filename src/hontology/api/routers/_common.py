"""Lookups and responses more than one router needs."""

from __future__ import annotations

from fastapi import HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from hontology.db.models import Run


def run_or_404(db: Session, run_id: int) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id} does not exist")
    return run


def truth_or_404(db: Session, ontology_id: int, annotator: str | None):
    """The labels scores are computed against: human for None, else that machine
    annotation set; 404 for a set that does not exist."""
    from hontology.retrieve import tuning

    try:
        return tuning.truth(db, ontology_id, annotator)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def csv_response(body: str, filename: str) -> PlainTextResponse:
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
