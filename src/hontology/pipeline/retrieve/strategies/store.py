"""Storing a run's retrievals, loading them back, and copying them to another run.

A retrieval is saved when an article is first retrieved, before any deciding
call, so a run that stops after retrieving resumes without retrieving again.
Its offers become `ShownLeaf` rows, one per leaf per deciding call: the shown
record. A flat prompt has no retrieval steps of its own; its deciding calls
are recorded as offers as they are made (`record_offer`).
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import ArticleRetrieval, Document, ShownLeaf
from hontology.pipeline.judge.providers.base import Completion
from hontology.pipeline.retrieve.strategies.base import Article, Asked, Offer, Retrieval
from hontology.pipeline.retrieve.strategies.extraction import Event

EMBEDDING = "embedding"


def save(session: Session, run_id: int, retrieval: Retrieval) -> ArticleRetrieval:
    """Add *retrieval* and its shown leaves to the session; the caller commits."""
    document_id = retrieval.article.document.id
    spend = retrieval.spend
    row = ArticleRetrieval(
        run_id=run_id,
        document_id=document_id,
        strategy=retrieval.strategy,
        trace=retrieval.trace,
        asked=[_asked_out(a) for a in retrieval.asked],
        events=[_event_out(e) for e in retrieval.events],
        parent_yes={str(c): yes for c, yes in retrieval.parent_yes.items()},
        stopped=retrieval.stopped,
        error=retrieval.error,
        input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens,
        calls=spend.calls,
        seconds=spend.seconds,
    )
    session.add(row)
    session.add_all(
        ShownLeaf(
            run_id=run_id,
            document_id=document_id,
            concept_id=concept_id,
            offer=ordinal,
            event=offer.event,
        )
        for ordinal, offer in enumerate(retrieval.offers)
        for concept_id in offer.concept_ids
    )
    return row


def load(
    session: Session, run_id: int, document: Document, body: str | None
) -> Retrieval | None:
    """*document*'s stored retrieval in *run_id*, or None if it has none."""
    row = session.scalar(
        select(ArticleRetrieval).where(
            ArticleRetrieval.run_id == run_id, ArticleRetrieval.document_id == document.id
        )
    )
    if row is None:
        return None
    retrieval = Retrieval(
        Article(document, body),
        strategy=row.strategy,
        asked=[_asked_in(a) for a in row.asked],
        events=[_event_in(e) for e in row.events],
        parent_yes={int(c): yes for c, yes in row.parent_yes.items()},
        trace=list(row.trace),
        stopped=row.stopped,
        error=row.error,
    )
    spend = retrieval.spend
    spend.input_tokens, spend.output_tokens = row.input_tokens, row.output_tokens
    spend.calls, spend.seconds = row.calls, row.seconds
    offers: dict[int, Offer] = {}
    for leaf in session.scalars(
        select(ShownLeaf)
        .where(ShownLeaf.run_id == run_id, ShownLeaf.document_id == document.id)
        .order_by(ShownLeaf.offer, ShownLeaf.concept_id)
    ):
        offers.setdefault(leaf.offer, Offer([], leaf.event)).concept_ids.append(leaf.concept_id)
    retrieval.offers = [offers[n] for n in sorted(offers)]
    return retrieval


def record_offer(
    session: Session, run_id: int, document_id: int, concept_ids: Iterable[int]
) -> None:
    """A flat prompt's deciding call, recorded as the article's next offer."""
    exists = session.scalar(
        select(ArticleRetrieval.id).where(
            ArticleRetrieval.run_id == run_id, ArticleRetrieval.document_id == document_id
        )
    )
    if exists is None:
        session.add(
            ArticleRetrieval(
                run_id=run_id, document_id=document_id, strategy=EMBEDDING, trace=[]
            )
        )
    ordinal = session.scalar(
        select(func.coalesce(func.max(ShownLeaf.offer) + 1, 0)).where(
            ShownLeaf.run_id == run_id, ShownLeaf.document_id == document_id
        )
    )
    session.add_all(
        ShownLeaf(run_id=run_id, document_id=document_id, concept_id=c, offer=ordinal)
        for c in concept_ids
    )


def copy(session: Session, source_run_id: int, target_run_id: int, document_ids=None) -> int:
    """Copy *source_run_id*'s stored retrievals the target lacks; returns how many.

    *document_ids* narrows the copy to some articles, as a calendar window does.
    """
    have = set(
        session.scalars(
            select(ArticleRetrieval.document_id).where(ArticleRetrieval.run_id == target_run_id)
        )
    )
    query = select(ArticleRetrieval).where(ArticleRetrieval.run_id == source_run_id)
    if document_ids is not None:
        query = query.where(among(ArticleRetrieval.document_id, document_ids))
    rows = [r for r in session.scalars(query) if r.document_id not in have]
    if not rows:
        return 0
    for row in rows:
        session.add(
            ArticleRetrieval(
                run_id=target_run_id,
                document_id=row.document_id,
                **{name: getattr(row, name) for name in _COPIED},
                copied_from_run=row.copied_from_run or source_run_id,
            )
        )
    for leaf in session.scalars(
        select(ShownLeaf).where(
            ShownLeaf.run_id == source_run_id,
            among(ShownLeaf.document_id, [r.document_id for r in rows]),
        )
    ):
        session.add(
            ShownLeaf(
                run_id=target_run_id,
                document_id=leaf.document_id,
                concept_id=leaf.concept_id,
                offer=leaf.offer,
                event=leaf.event,
            )
        )
    session.flush()
    return len(rows)


_COPIED = (
    "strategy",
    "trace",
    "asked",
    "events",
    "parent_yes",
    "stopped",
    "error",
    "input_tokens",
    "output_tokens",
    "calls",
    "seconds",
)


def _asked_out(asked: Asked) -> dict:
    completion = asked.completion
    return {
        "concept_ids": asked.concept_ids,
        "replies": None
        if asked.replies is None
        else {str(c): reply for c, reply in asked.replies.items()},
        "error": asked.error,
        "completion": None
        if completion is None
        else {
            "reasoning": completion.reasoning,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_s": completion.latency_s,
        },
    }


def _asked_in(stored: dict) -> Asked:
    replies = stored["replies"]
    completion = stored["completion"]
    return Asked(
        stored["concept_ids"],
        None if replies is None else {int(c): reply for c, reply in replies.items()},
        None if completion is None else Completion(text="", **completion),
        stored["error"],
    )


def _event_out(event: Event) -> dict:
    return {
        "found": event.found,
        "routed": event.routed,
        "reached": sorted(event.reached),
        "error": event.error,
    }


def _event_in(stored: dict) -> Event:
    return Event(stored["found"], stored["routed"], set(stored["reached"]), stored["error"])
