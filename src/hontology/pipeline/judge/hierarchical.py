"""Hierarchical judging: route each document down the hierarchy, then decide.

A document is judged if retrieval selected at least one leaf for it, the same
gate the flat arm passes through. Routing (`pipeline.retrieve.strategies.routing`)
then asks the parent classes and offers the leaves they lead to, one set per
parent; each offer is one deciding call, asked exactly as the flat arm asks.

- **Routing answers are verdicts too.** Each parent gets a row with its
  routing call's answer and cost, so recall per level can be read back.
- **Leaves never offered get no row.** Metrics read them as negatives.
- **Stored before deciding.** Each document's retrieval is saved, with its
  routing answers, before any leaf is decided. A restart, or a run that
  adopted another's retrieval, decides from the stored one without routing
  again, and an offer whose leaves are all stored is not decided again.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from hontology.db.models import Candidate, Document, Verdict
from hontology.ontology import hierarchy
from hontology.pipeline.judge.batched import judge_call, record_batch
from hontology.pipeline.judge.context import Judging
from hontology.pipeline.retrieve import strategies
from hontology.pipeline.retrieve.strategies import store

log = logging.getLogger(__name__)


def judge_hierarchical(
    judging: Judging,
    *,
    ontology_id: int,
    candidates: list[Candidate],
    limit: int | None,
    progress: object | None,
) -> dict:
    """Route and judge each gated document. See the module docstring."""
    session, template, stats = judging.session, judging.template, judging.stats
    children = hierarchy.children(session, ontology_id)
    top = sorted(hierarchy.top_level(session, ontology_id))
    documents = sorted({c.document_id for c in candidates})
    stats.total = len(documents)
    processed = calls = omitted_total = 0

    for done_documents, document_id in enumerate(documents, start=1):
        if limit is not None and processed >= limit:
            break
        document = session.get(Document, document_id)
        if document is None:
            continue
        stored = _stored(judging, document_id)
        body = judging.body(document)
        retrieval = store.load(session, judging.run_id, document, body)
        fresh = retrieval is None
        if retrieval is None:
            known = {c: yes for c, yes in stored.items() if children.get(c)}
            strategy = strategies.for_template(judging, top=top, children=children, known=known)
            retrieval = strategy.retrieve(strategies.Article(document, body))
            store.save(session, judging.run_id, retrieval)

        if retrieval.error:
            unasked = [c for c in top if c not in stored]
            if unasked:
                session.add_all(judging.failed(document_id, unasked, retrieval.error))
            session.commit()
            continue
        # Routing answers, unless already written: in the commit that saved the
        # retrieval, or by an older run that stored no retrieval.
        for asked in retrieval.asked:
            if any(c in stored for c in asked.concept_ids):
                continue
            _, omitted = record_batch(judging, document_id, asked)
            omitted_total += omitted
            processed += len(asked.concept_ids)
            calls += int(fresh)
        session.commit()
        assert body is not None
        for offer in retrieval.offers:
            if all(c in stored for c in offer.concept_ids):
                continue
            _, omitted = judge_call(
                judging, document, judging.concepts(offer.concept_ids), body
            )
            omitted_total += omitted
            processed += len(offer.concept_ids)
            calls += 1
            session.commit()

        if callable(progress):
            progress(done_documents, len(documents))

    result = stats.as_dict() | {
        "mode": template.mode,
        "documents": len(documents),
        "calls": calls,
        "omitted_by_model": omitted_total,
    }
    log.info("judge (hierarchical): %s", result)
    return result


def _stored(judging: Judging, document_id: int) -> dict[int, bool | None]:
    """What is already stored for *document_id*, so a restart resumes mid-document."""
    return {
        concept_id: (None if error else matched)
        for concept_id, matched, error in judging.session.execute(
            select(Verdict.concept_id, Verdict.matched, Verdict.error).where(
                Verdict.run_id == judging.run_id, Verdict.document_id == document_id
            )
        )
    }
