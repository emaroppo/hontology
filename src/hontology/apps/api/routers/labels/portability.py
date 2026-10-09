"""Label bank portability.

The bank is hours of human attention, so it has to travel: reviewable in a
diff, safe to re-import, and keyed by names rather than local database ids.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hontology.apps.api.routers._common import csv_response
from hontology.db.session import get_db
from hontology.evaluation.labels import csv_export, csv_io

router = APIRouter(prefix="/labels", tags=["labels"])


class ImportIn(BaseModel):
    ontology_id: int
    csv: str
    # Off by default: importing must never silently destroy adjudicated work.
    overwrite: bool = False
    create_missing_documents: bool = True


@router.get("/export", response_class=PlainTextResponse)
def export_labels(ontology_id: int, db: Session = Depends(get_db)):
    """All pair labels as CSV. Stale ones are included and flagged, not dropped."""
    return csv_response(csv_export.export_labels(db, ontology_id), f"labels-{ontology_id}.csv")


@router.get("/observations/export", response_class=PlainTextResponse)
def export_observations(ontology_id: int, db: Session = Depends(get_db)):
    """Known occurrences as CSV."""
    return csv_response(
        csv_export.export_observations(db, ontology_id), f"observations-{ontology_id}.csv"
    )


@router.post("/import")
def import_labels(payload: ImportIn, db: Session = Depends(get_db)):
    """Load labels from CSV, matching on document URL and concept name."""
    return csv_io.import_labels(
        db,
        payload.ontology_id,
        payload.csv,
        overwrite=payload.overwrite,
        create_missing_documents=payload.create_missing_documents,
    )


@router.post("/observations/import")
def import_observations(payload: ImportIn, db: Session = Depends(get_db)):
    """Load known occurrences from CSV."""
    return csv_io.import_observations(db, payload.ontology_id, payload.csv)
