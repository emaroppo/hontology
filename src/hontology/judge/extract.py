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

import numpy as np
from sqlalchemy import select

from hontology.db.models import Candidate, Concept, Document, ExtractedEvent, Verdict
from hontology.judge import prompts
from hontology.judge.context import NO_BODY, Judging, share
from hontology.judge.parse import parse_batch, parse_choice, parse_events
from hontology.judge.providers.base import Completion, ProviderError
from hontology.ontology import hierarchy
from hontology.retrieve import embed

log = logging.getLogger(__name__)


class _LeafRanker:
    """Orders leaves by how close their embedding is to an event's."""

    def __init__(self, provider, model: str, leaves: list[Concept], k: int) -> None:
        self.provider, self.model, self.k = provider, model, k
        self.ids = [c.id for c in leaves]
        prefix = embed.document_prefix(model)
        self.vectors = self._unit(
            [prefix + embed.concept_text(c, "name+definition") for c in leaves]
        )

    def _unit(self, texts: list[str]) -> np.ndarray:
        texts = [embed.normalize_for_embedding(t) for t in texts]
        vectors = np.array(self.provider.embed(texts, model=self.model), dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def __call__(self, event: dict, allowed: set[int]) -> list[int]:
        text = f"{event['description']}\n{event['evidence']}".strip()
        scores = self.vectors @ self._unit([embed.query_prefix(self.model) + text])[0]
        ranked = sorted(
            (i for i in range(len(self.ids)) if self.ids[i] in allowed),
            key=lambda i: (-scores[i], self.ids[i]),
        )
        return [self.ids[i] for i in ranked[: self.k]]


class _Spend:
    """What one article's calls cost, to be split across its verdict rows."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0
        self.calls = 0

    def add(self, completion: Completion) -> Completion:
        self.input_tokens += completion.input_tokens or 0
        self.output_tokens += completion.output_tokens or 0
        self.seconds += completion.latency_s or 0.0
        self.calls += 1
        return completion


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
        rank = _LeafRanker(
            embedder,
            template.leaf_embed_model,
            [concept(c) for c in leaves],
            template.leaf_top_k,
        )

    def call(system: str | None, prompt: str, spend: _Spend) -> Completion:
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
        spend = _Spend()
        body = judging.body(document)
        rows: list[Verdict] = []
        event_rows: list[ExtractedEvent] = []
        if not body:
            rows = judging.failed(document_id, top, NO_BODY)
        else:
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
                rows = judging.failed(
                    document_id, top, f"event extraction failed: {exc}"[:1000]
                )
                events = None
            if events is not None:
                rows, event_rows = _classify(
                    judging,
                    events,
                    document_id=document_id,
                    top=top,
                    children=children,
                    concept=concept,
                    call=call,
                    spend=spend,
                    rank=rank,
                )
                events_total += len(events)

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


def _classify(
    judging: Judging,
    events: list[dict],
    *,
    document_id: int,
    top: list[int],
    children: dict[int, set[int]],
    concept,
    call,
    spend: _Spend,
    rank=None,
) -> tuple[list[Verdict], list[ExtractedEvent]]:
    """Route each event down the hierarchy and choose at most one leaf for it."""
    template = judging.template
    assert template.build_event_route is not None and template.build_choose is not None
    parent_yes: dict[int, bool] = {}
    weighed: set[int] = set()
    chosen: dict[int, tuple[float, str, str]] = {}  # leaf -> confidence, evidence, country
    event_rows: list[ExtractedEvent] = []

    for ordinal, event in enumerate(events):
        routed: list[int] = []
        reached: set[int] = set()
        choice: int | None = None
        confidence: float | None = None
        error: str | None = None
        outcome = "rejected"
        try:
            frontier, seen, first = list(top), set(), True
            while frontier:
                seen |= set(frontier)
                parents = [c for c in frontier if children.get(c)]
                # At the top every class is a routing question, leaves included,
                # beside "other"; below, a routed parent's leaves are candidates.
                asked = frontier if first else parents
                if not first:
                    reached |= {c for c in frontier if not children.get(c)}
                below: set[int] = set()
                if asked:
                    reply = call(
                        template.event_top_system if first else template.event_route_system,
                        template.build_event_route(
                            event, [concept(c) for c in asked], other=first
                        ),
                        spend,
                    )
                    expected = [*asked, prompts.OTHER_ID] if first else asked
                    answers = parse_batch(reply.text, expected)
                    for c in asked:
                        yes = bool(answers[c]["matched"])
                        if children.get(c):
                            parent_yes[c] = parent_yes.get(c, False) or yes
                            if yes:
                                routed.append(c)
                                below |= set(children[c])
                        elif yes:
                            reached.add(c)
                if first and rank is not None:
                    # Below the top, the closest leaves under the classes answered
                    # yes are the candidates, with no further routing.
                    allowed = reached | _leaves_under(routed, children)
                    reached = set(rank(event, allowed)) if allowed else set()
                    break
                first = False
                frontier = sorted(below - seen)
            # Nothing but "other" took it: rejected at the top, no choosing call.
            if reached:
                weighed |= reached
                reply = call(
                    template.choose_system,
                    template.build_choose(event, [concept(c) for c in sorted(reached)]),
                    spend,
                )
                choice, confidence, quote = parse_choice(reply.text, reached)
                outcome = "classified" if choice is not None else "unclassified"
                if choice is not None and (
                    choice not in chosen or confidence > chosen[choice][0]
                ):
                    chosen[choice] = (confidence, quote or event["evidence"], event["country"])
        except (ProviderError, json.JSONDecodeError) as exc:
            error = str(exc)[:1000]
            outcome = "error"
            judging.stats.errors += 1
        event_rows.append(
            ExtractedEvent(
                run_id=judging.run_id,
                document_id=document_id,
                ordinal=ordinal,
                description=event["description"],
                evidence=event["evidence"] or None,
                status=event["status"] or None,
                country=event["country"] or None,
                concept_id=choice,
                routed_through=routed,
                confidence=confidence,
                outcome=outcome,
                error=error,
            )
        )

    rows = [
        judging.verdict(document_id, c, matched=yes) for c, yes in sorted(parent_yes.items())
    ]
    for leaf in sorted(weighed):
        confidence, evidence, country = chosen.get(leaf, (None, "", ""))
        rows.append(
            judging.verdict(
                document_id,
                leaf,
                matched=leaf in chosen,
                confidence=confidence,
                evidence=evidence or None,
                locus_id=judging.iso2_to_locus.get(country) if country else None,
            )
        )
    if not rows:
        # No events, or none reached anything: the article is a no at the top,
        # and these rows carry what the extraction cost.
        rows = [
            judging.verdict(
                document_id,
                c,
                matched=False,
                evidence="no events extracted" if not events else None,
            )
            for c in top
        ]
    return rows, event_rows


def _leaves_under(classes: list[int], children: dict[int, set[int]]) -> set[int]:
    out: set[int] = set()
    stack = list(classes)
    while stack:
        c = stack.pop()
        if children.get(c):
            stack.extend(children[c])
        else:
            out.add(c)
    return out
