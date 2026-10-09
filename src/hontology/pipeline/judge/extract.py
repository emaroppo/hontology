"""Extract-then-classify judging (arm E): events first, one class per event.

Judging a whole article top-down lets one event be ticked under several sibling
classes: a strike on a pipeline becomes a shutdown, a production halt and an
industrial accident. Here each article is read once, by a call that lists its
distinct events; each event is then routed down the class hierarchy on its own
description and evidence, and given **at most one leaf**. An article reporting
several events (a tariff imposed and further tariffs threatened) keeps a class
for each.

An extract template may replace the routing below the top level
(``leaf_top_k``): an event the top level accepts is then offered the
``leaf_top_k`` leaves, under the classes it answered yes to, whose embeddings are
closest to its own, and goes straight to the choosing call.

At the top, each event is also asked whether it is "other": none of the
classes, under an instruction that leans neither way; below the top, routing
leans to yes, as in hierarchical judging. Top-level leaves are questions at the
top too. An event nothing but
"other" accepts is rejected with no further calls, which is what most events in
a news article are; each event's outcome (classified, unclassified, rejected,
error) is recorded with it.

The article's verdicts are derived from its events:

- a parent class is yes if any event was routed through it;
- a leaf any event was weighed against is yes if an event chose it, no if none
  did; leaves no event reached get no row, and score as no;
- an article with no events gets a no for each top-level class, which also
  carries the extraction call's cost.

Every call's tokens are split exactly across the article's verdict rows, so a
run's recorded cost is what it spent. An article is written whole, with its
events, in one commit; a restart skips articles already written.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select

from hontology.db.models import Candidate, Concept, Document, ExtractedEvent, Verdict
from hontology.ontology import hierarchy
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.context import NO_BODY, Judging, share
from hontology.pipeline.judge.events import Call, LeafRanker, Spend, classify_events
from hontology.pipeline.judge.parse import parse_events
from hontology.pipeline.judge.providers.base import Completion, ProviderError

log = logging.getLogger(__name__)


def judge_extract(
    judging: Judging,
    *,
    ontology_id: int,
    candidates: list[Candidate],
    limit: int | None,
    progress: object | None,
    embedder=None,
) -> dict:
    """Judge each gated document by its events. See the module docstring."""
    session, template, stats = judging.session, judging.template, judging.stats
    assert template.mode == prompts.EXTRACT
    children = hierarchy.children(session, ontology_id)
    top = sorted(hierarchy.top_level(session, ontology_id))
    concepts: dict[int, Concept] = {}

    def concept(class_id: int) -> Concept:
        if class_id not in concepts:
            found = session.get(Concept, class_id)
            assert found is not None
            concepts[class_id] = found
        return concepts[class_id]

    rank = None
    if template.leaf_top_k is not None:
        assert template.leaf_embed_model is not None and embedder is not None
        leaves = sorted(hierarchy.leaves(session, ontology_id))
        rank = LeafRanker(
            embedder,
            template.leaf_embed_model,
            [concept(c) for c in leaves],
            template.leaf_top_k,
        )

    def call(system: str | None, prompt: str, spend: Spend) -> Completion:
        return spend.add(judging.ask(system, prompt))

    documents = sorted({c.document_id for c in candidates})
    written = set(
        session.scalars(
            select(Verdict.document_id).where(Verdict.run_id == judging.run_id).distinct()
        )
    )
    stats.total = len(documents)
    calls = events_total = processed = 0

    for done_documents, document_id in enumerate(documents, start=1):
        if document_id in written:
            stats.skipped += 1
            continue
        if limit is not None and processed >= limit:
            break
        document = session.get(Document, document_id)
        if document is None:
            continue
        processed += 1
        spend = Spend()
        rows, event_rows = _judge_document(
            judging,
            document,
            top=top,
            children=children,
            concept=concept,
            call=call,
            spend=spend,
            rank=rank,
        )
        events_total += len(event_rows)

        for position, row in enumerate(rows):
            row.input_tokens = share(spend.input_tokens, len(rows), position)
            row.output_tokens = share(spend.output_tokens, len(rows), position)
            row.latency_s = spend.seconds / len(rows) if rows else None
        session.add_all(event_rows)
        session.add_all(rows)
        session.commit()
        stats.judged += sum(1 for r in rows if r.error is None)
        stats.matched += sum(1 for r in rows if r.matched)
        stats.input_tokens += spend.input_tokens
        stats.output_tokens += spend.output_tokens
        calls += spend.calls
        if callable(progress):
            progress(done_documents, len(documents))

    result = stats.as_dict() | {
        "mode": template.mode,
        "documents": len(documents),
        "calls": calls,
        "events": events_total,
    }
    log.info("judge (extract): %s", result)
    return result


def _judge_document(
    judging: Judging,
    document: Document,
    *,
    top: list[int],
    children: dict[int, set[int]],
    concept,
    call: Call,
    spend: Spend,
    rank: LeafRanker | None,
) -> tuple[list[Verdict], list[ExtractedEvent]]:
    """List one article's events, then classify each: its verdicts and events."""
    template = judging.template
    body = judging.body(document)
    if not body:
        return judging.failed(document.id, top, NO_BODY), []
    try:
        assert template.build_extract is not None
        reply = call(
            template.extract_system,
            template.build_extract(
                document, [concept(c) for c in top], body, judging.body_limit
            ),
            spend,
        )
        events = parse_events(reply.text)
    except (ProviderError, json.JSONDecodeError) as exc:
        return judging.failed(document.id, top, f"event extraction failed: {exc}"[:1000]), []
    return classify_events(
        judging,
        events,
        document_id=document.id,
        top=top,
        children=children,
        concept=concept,
        call=call,
        spend=spend,
        rank=rank,
    )
