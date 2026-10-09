"""Small query helpers: counts, id <-> name maps over concepts and loci, a run by id."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Locus, Run


def count(session: Session, expression: Any, *where: Any) -> int:
    """``SELECT count(expression) WHERE ...``, zero when there is nothing."""
    return session.scalar(select(func.count(expression)).where(*where)) or 0


def concepts_by_id(session: Session, ontology_id: int) -> dict[int, Concept]:
    return {
        c.id: c
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }


def concept_names(session: Session, ontology_id: int) -> dict[int, str]:
    return {c.id: c.name for c in concepts_by_id(session, ontology_id).values()}


def concept_ids_by_name(session: Session, ontology_id: int) -> dict[str, int]:
    return {c.name: c.id for c in concepts_by_id(session, ontology_id).values()}


def locus_iso3s(session: Session) -> dict[int, str | None]:
    return {locus.id: locus.iso3 for locus in session.scalars(select(Locus))}


def locus_ids_by_iso3(session: Session) -> dict[str, int]:
    return {
        locus.iso3: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso3.is_not(None)))
    }


def get_run(session: Session, run_id: int) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")
    return run
