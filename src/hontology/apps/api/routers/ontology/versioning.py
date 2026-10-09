"""Ontology versioning: snapshots, minted versions, and the lint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.apps.api.routers.ontology.lookups import ontology_or_404
from hontology.apps.api.schemas.ontology import SnapshotOut
from hontology.db.models import OntologySnapshot
from hontology.db.session import get_db
from hontology.ontology import snapshots

router = APIRouter(prefix="/ontologies", tags=["ontology"])


@router.post("/{ontology_id}/snapshot", response_model=SnapshotOut)
def resolve_snapshot(ontology_id: int, db: Session = Depends(get_db)):
    """Resolve the live concept set to a version, minting one only if it changed.

    Safe to call repeatedly: an unchanged ontology keeps returning the same
    version with ``created=false``.
    """
    ontology_or_404(db, ontology_id)
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
    ontology_or_404(db, ontology_id)
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
    from hontology.ontology.lint import report as lint_module

    ontology_or_404(db, ontology_id)
    return lint_module.lint(db, ontology_id)
