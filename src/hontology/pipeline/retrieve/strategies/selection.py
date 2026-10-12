"""The embedding cutoff as a step: the leaves selected for an article.

For a flat strategy this is the whole retrieval, and the selected leaves are
the offer. For a top-down strategy it is the gate: an article with no leaf
selected stops here, and later steps decide what is shown.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate
from hontology.pipeline.retrieve.strategies.base import Article, Offer, Retrieval

GATE_CLOSED = "no leaf selected"


@dataclass
class Selected:
    choose: Callable[[Article], list[int]]
    # Flat: the selected leaves are the offer. Top-down: they only gate.
    offer: bool
    name: str = "embedding"

    def __call__(self, retrieval: Retrieval) -> None:
        leaves = sorted(self.choose(retrieval.article))
        retrieval.note(self.name, selected=leaves)
        if not leaves:
            retrieval.stopped = GATE_CLOSED
        elif self.offer:
            retrieval.offers.append(Offer(leaves))


def stored(session: Session, run_id: int) -> Callable[[Article], list[int]]:
    """The leaves a run's stored retrieval selected for an article."""

    def choose(article: Article) -> list[int]:
        return list(
            session.scalars(
                select(Candidate.concept_id).where(
                    Candidate.run_id == run_id,
                    Candidate.document_id == article.document.id,
                    Candidate.selected.is_(True),
                )
            )
        )

    return choose
