"""Retrieval reuse: adopting another run's candidates instead of recomputing them.

Two runs sharing a ``candidates_key`` share retrieval, so the second copies the
first's rows, which is what makes iterating on a prompt cost only the judging.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Run


def reusable_candidates_run(session: Session, run: Run) -> Run | None:
    """A completed earlier run whose retrieval this one can adopt verbatim."""
    return session.scalar(
        select(Run)
        .where(
            Run.id != run.id,
            Run.ontology_id == run.ontology_id,
            Run.candidates_key == run.candidates_key,
            Run.status.in_(("done", "judged", "candidates")),
        )
        .order_by(Run.id)
        .limit(1)
    )


def clone_candidates(session: Session, rows: Iterable[Candidate], target_run_id: int) -> int:
    """Add a copy of each candidate under *target_run_id*; returns how many."""
    copied = 0
    for row in rows:
        session.add(
            Candidate(
                run_id=target_run_id,
                document_id=row.document_id,
                concept_id=row.concept_id,
                source=row.source,
                score=row.score,
                rank=row.rank,
                selected=row.selected,
                matched_code=row.matched_code,
                matched_level=row.matched_level,
            )
        )
        copied += 1
    session.flush()
    return copied


def copy_candidates(session: Session, source_run_id: int, target_run_id: int) -> int:
    rows = list(session.scalars(select(Candidate).where(Candidate.run_id == source_run_id)))
    return clone_candidates(session, rows, target_run_id)
