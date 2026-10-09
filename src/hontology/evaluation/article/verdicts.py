"""A run's answers on labelled pairs: what it judged, and what it said."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import ScalarResult, select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Verdict
from hontology.evaluation.metrics.confusion import Confusion

Key = tuple[int, int]


def clean_verdicts(session: Session, run_id: int) -> ScalarResult[Verdict]:
    """A run's verdicts that carry an answer: no error, and a match decided."""
    return session.scalars(
        select(Verdict).where(
            Verdict.run_id == run_id, Verdict.error.is_(None), Verdict.matched.is_not(None)
        )
    )


def answered(
    session: Session, run_id: int, documents: Iterable[int] | None = None
) -> dict[Key, bool]:
    """The run's clean answer on each pair it judged, optionally for some documents."""
    query = select(Verdict.document_id, Verdict.concept_id, Verdict.matched).where(
        Verdict.run_id == run_id, Verdict.error.is_(None), Verdict.matched.is_not(None)
    )
    if documents is not None:
        query = query.where(among(Verdict.document_id, documents))
    return {
        (doc_id, concept_id): bool(matched)
        for doc_id, concept_id, matched in session.execute(query)
    }


def predictions(session: Session, run_id: int, keys: set[Key]) -> dict[Key, bool]:
    """The run's answer on each labelled pair; unjudged and errored pairs are no."""
    judged = answered(session, run_id, {doc_id for doc_id, _ in keys})
    return {key: judged.get(key, False) for key in keys}


def judged_keys(session: Session, run_id: int, keys: set[Key]) -> set[Key]:
    return keys & answered(session, run_id, {doc_id for doc_id, _ in keys}).keys()


def confusion(truth: dict[Key, bool], predicted: dict[Key, bool], keys=None) -> Confusion:
    result = Confusion()
    for key in keys if keys is not None else truth:
        result.add(expected=truth[key], predicted=predicted.get(key, False))
    return result
