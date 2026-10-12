"""What a run showed its judge: the one definition of *retrieved* every metric uses.

A (document, leaf) pair is retrieved if and only if the leaf was in a deciding
judge call for that document: it has a `ShownLeaf` row. Routing calls and the
embedding cutoff are how a pair comes to be shown, not what retrieval is.

The judge is scored on the pairs it was shown and nothing else. A shown pair
whose call failed counts as a "no" from the judge, as end to end already does,
so a judge that errors cannot raise its recall by dropping pairs; liveness
reports the failures. A positive never shown is retrieval's miss, in every arm.

On the labelled positives of the articles a run covered, end-to-end recall is
then retrieval recall times judge recall.

Runs made before the shown record existed have none until they are backfilled;
`has_record` tells them apart from a run that showed nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import ArticleRetrieval, Candidate, ShownLeaf, Verdict

Key = tuple[int, int]


def has_record(session: Session, run_id: int) -> bool:
    """Whether *run_id* stored what it showed its judge."""
    return (
        session.scalar(
            select(ArticleRetrieval.id).where(ArticleRetrieval.run_id == run_id).limit(1)
        )
        is not None
    )


def shown_pairs(
    session: Session, run_id: int, documents: Iterable[int] | None = None
) -> set[Key]:
    """The (document, leaf) pairs *run_id* showed a deciding call."""
    query = select(ShownLeaf.document_id, ShownLeaf.concept_id).where(
        ShownLeaf.run_id == run_id
    )
    if documents is not None:
        query = query.where(among(ShownLeaf.document_id, set(documents)))
    return set(session.execute(query.distinct()).tuples())


def covered_documents(session: Session, run_id: int) -> set[int]:
    """The articles *run_id* ranked: where a positive could have been shown."""
    return set(
        session.scalars(
            select(Candidate.document_id).where(Candidate.run_id == run_id).distinct()
        )
    )


def judge_answers(session: Session, run_id: int, shown: set[Key]) -> dict[Key, bool]:
    """The judge's answer on each shown pair: its verdict, or "no" where the
    call failed or left the pair out."""
    answered = {
        (document_id, concept_id): bool(matched)
        for document_id, concept_id, matched in session.execute(
            select(Verdict.document_id, Verdict.concept_id, Verdict.matched).where(
                Verdict.run_id == run_id,
                Verdict.error.is_(None),
                Verdict.matched.is_not(None),
                among(Verdict.document_id, {doc for doc, _ in shown}),
            )
        )
    }
    return {key: answered.get(key, False) for key in shown}


def retrieval_scores(session: Session, run_id: int, truth: dict[Key, bool]) -> dict:
    """How much of the labelled truth *run_id* showed its judge.

    Recall is over the labelled positives on the articles the run ranked, the
    only ones it could have shown; precision is over the shown pairs that carry
    a label, and ``coverage`` says how many of them did.
    """
    from hontology.evaluation.metrics.intervals import wilson

    covered = covered_documents(session, run_id)
    shown = shown_pairs(session, run_id, covered)
    positives = {key for key, matched in truth.items() if matched and key[0] in covered}
    labelled = {key for key in shown if key in truth}
    hits = len(positives & shown)
    true_shown = sum(1 for key in labelled if truth[key])
    return {
        "shown_record": has_record(session, run_id),
        "shown_pairs": len(shown),
        "labelled_shown": len(labelled),
        "positives": len(positives),
        "positives_shown": hits,
        "recall": hits / len(positives) if positives else None,
        "recall_ci": wilson(hits, len(positives)),
        "precision": true_shown / len(labelled) if labelled else None,
        "precision_ci": wilson(true_shown, len(labelled)),
        "coverage": len(labelled) / len(shown) if shown else None,
    }
