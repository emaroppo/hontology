"""One run scored end to end on a labelled sample."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hontology.apps.api.routers._common import run_or_404, truth_or_404
from hontology.db.session import get_db

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/runs/{run_id}/sample")
def run_on_sample(run_id: int, manifest: str, db: Session = Depends(get_db)):
    """One run's end-to-end scores on a labelled sample, with the pairs it got wrong.

    *manifest* is the sample manifest's path on the API's machine.
    """
    return _sample(db, run_id, manifest, None)


class SampleIn(BaseModel):
    manifest_path: str
    # The truth: None for human labels, else a machine annotation set's name.
    annotator: str | None = None


@router.post("/runs/{run_id}/sample")
def run_on_sample_with(run_id: int, payload: SampleIn, db: Session = Depends(get_db)):
    """As the GET, scored against a machine annotation set if one is named."""
    return _sample(db, run_id, payload.manifest_path, payload.annotator)


def read_manifest(path: str) -> dict:
    import json
    from pathlib import Path

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"cannot read manifest: {exc}"
        ) from exc


def sample_labels(db: Session, ontology_id: int, annotator: str | None) -> dict | None:
    """None for human labels, which the sample scoring reads from the bank itself."""
    return None if annotator is None else truth_or_404(db, ontology_id, annotator)


def _sample(db: Session, run_id: int, manifest_path: str, annotator: str | None) -> dict:
    from hontology.db.models import Concept, Document
    from hontology.evaluation.comparison import arms

    record = read_manifest(manifest_path)
    run = run_or_404(db, run_id)
    labels = sample_labels(db, run.ontology_id, annotator)
    result = arms.run_on_sample(db, run_id, record, labels=labels)
    for error in result.get("errors", []):
        concept = db.get(Concept, error["concept_id"])
        document = db.get(Document, error["document_id"])
        error["concept"] = concept.name if concept else None
        error["document_url"] = document.url if document else None
        error["document_title"] = document.title if document else None
    return result
