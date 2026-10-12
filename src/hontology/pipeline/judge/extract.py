"""Extract-then-classify judging (arm E): events first, one class per event.

Judging a whole article top-down lets one event be ticked under several sibling
classes: a strike on a pipeline becomes a shutdown, a production halt and an
industrial accident. Here retrieval (`pipeline.retrieve.strategies.extraction`)
lists the article's distinct events and routes each down the class hierarchy,
offering the leaves it reaches; the judge then gives each event **at most one
leaf** in a choosing call. An article reporting several events (a tariff
imposed and further tariffs threatened) keeps a class for each.

The article's verdicts are derived from its events:

- a parent class is yes if any event was routed through it;
- a leaf any event was offered is yes if an event chose it, no if none did;
  leaves no event reached get no row, and score as no;
- an article with no events gets a no for each top-level class, which also
  carries the extraction call's cost.

Every call's tokens, retrieval's included, are split exactly across the
article's verdict rows, so a run's recorded cost is what it spent. An article's
retrieval is saved before its choosing calls, and its verdicts and events are
written whole in one commit after them: a restart skips articles already
written, and chooses from the stored retrieval for one retrieved but not
written.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from hontology.db.models import Candidate, Document, ExtractedEvent, Verdict
from hontology.ontology import hierarchy
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.context import Judging, share
from hontology.pipeline.judge.events import decide_events
from hontology.pipeline.retrieve import strategies
from hontology.pipeline.retrieve.strategies import store

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
    ranker = None
    if template.leaf_top_k is not None:
        assert template.leaf_embed_model is not None and embedder is not None
        ranker = strategies.LeafRanker(
            embedder,
            template.leaf_embed_model,
            judging.concepts(sorted(hierarchy.leaves(session, ontology_id))),
            template.leaf_top_k,
        )
    strategy = strategies.for_template(judging, top=top, children=children, ranker=ranker)

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
        body = judging.body(document)
        retrieval = store.load(session, judging.run_id, document, body)
        if retrieval is None:
            retrieval = strategy.retrieve(strategies.Article(document, body))
            # Saved before choosing, so a restart chooses without extracting again.
            store.save(session, judging.run_id, retrieval)
            session.commit()
        if retrieval.error:
            rows = judging.failed(document_id, top, retrieval.error)
            event_rows: list[ExtractedEvent] = []
        else:
            rows, event_rows = decide_events(judging, retrieval, document_id, top)
        events_total += len(event_rows)

        spend = retrieval.spend
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
