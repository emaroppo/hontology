"""Judging several concepts per call: one call per document.

`ask_batch` is shared with routing (`pipeline.retrieve.strategies.routing`),
and `judge_call` with hierarchical judging (`pipeline.judge.hierarchical`).
"""

from __future__ import annotations

import logging

from hontology.db.models import Candidate, Concept, Document
from hontology.pipeline.judge.context import NO_BODY, Judging, share
from hontology.pipeline.judge.parse import parse_batch
from hontology.pipeline.retrieve.strategies.base import CALL_ERRORS, Asked, Model

log = logging.getLogger(__name__)


def ask_batch(
    model: Model,
    document: Document,
    concepts: list[Concept],
    body: str,
    *,
    routing: bool = False,
) -> Asked:
    """One batched call over *concepts*, nothing recorded.

    With *routing*, the concepts are parent classes asked as routing questions.
    """
    template = model.template
    if routing:
        assert template.build_route is not None and template.route_system is not None
        prompt = template.build_route(document, concepts, body, model.body_limit)
        system = template.route_system
    else:
        assert template.build_batch is not None  # guaranteed by PromptTemplate
        prompt = template.build_batch(document, concepts, body, model.body_limit)
        system = template.system
    ids = [c.id for c in concepts]
    try:
        completion = model.ask(system, prompt)
        return Asked(ids, parse_batch(completion.text, ids), completion)
    except CALL_ERRORS as exc:
        return Asked(ids, None, None, f"{type(exc).__name__}: {exc}")


def record_batch(judging: Judging, document_id: int, asked: Asked) -> tuple[dict, int]:
    """Write a verdict for each class of one batched call.

    Returns each class's answer (None when the call failed) and how many
    classes the model left out of its reply.
    """
    session, ids = judging.session, asked.concept_ids
    if asked.replies is None or asked.completion is None:
        # One bad response costs the whole set — recorded per pair so the
        # denominator stays honest and liveness catches it.
        session.add_all(judging.failed(document_id, ids, asked.error or "call failed"))
        return dict.fromkeys(ids), 0

    completion = asked.completion
    judging.stats.spent(completion)
    omitted = 0
    for position, concept_id in enumerate(ids):
        result = asked.replies[concept_id]
        if result.get("omitted"):
            omitted += 1
        session.add(
            judging.verdict(
                document_id,
                concept_id,
                matched=result["matched"],
                confidence=result["confidence"],
                vote_fraction=1.0,
                locus_id=judging.iso2_to_locus.get(result["country"]),
                evidence=result["evidence"] or None,
                reasoning=completion.reasoning or None,
                # One call served the whole set, so attributing its full cost
                # to each pair would multiply the real cost.
                latency_s=completion.latency_s / len(ids),
                input_tokens=share(completion.input_tokens, len(ids), position),
                output_tokens=share(completion.output_tokens, len(ids), position),
            )
        )
        judging.stats.judged += 1
        judging.stats.matched += int(result["matched"])
    return {c: asked.answer(c) for c in ids}, omitted


def judge_call(
    judging: Judging,
    document: Document,
    concepts: list[Concept],
    body: str,
) -> tuple[dict[int, bool | None], int]:
    """One batched deciding call over *concepts*, writing a verdict for each.

    Shared by batched and hierarchical judging, so both ask the judge in
    exactly the same way.
    """
    return record_batch(judging, document.id, ask_batch(judging, document, concepts, body))


def judge_batched(
    judging: Judging,
    *,
    candidates: list[Candidate],
    done: set[tuple[int, int]],
    limit: int | None,
    progress: object | None,
) -> dict:
    """One call per document, judging every candidate concept at once.

    The cost profile is the point: N concepts for one article cost one call and
    one copy of the body, rather than N calls each re-sending it. What is traded
    away is isolation — a single malformed response costs every pair for that
    document, not one — so a failure is recorded against each of them rather than
    losing them silently.

    Sampling does not apply. Repeating a batch call re-rolls every verdict
    together, so the votes are not independent and a vote fraction across them
    would overstate agreement. Batch runs are a single greedy call.
    """
    session, stats = judging.session, judging.stats
    by_document: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        if (candidate.document_id, candidate.concept_id) in done:
            stats.skipped += 1
            continue
        by_document.setdefault(candidate.document_id, []).append(candidate)

    processed = 0
    omitted_total = 0

    for document_id, group in by_document.items():
        if limit is not None and processed >= limit:
            break

        document = session.get(Document, document_id)
        if document is None:
            continue
        concepts = judging.concepts(c.concept_id for c in group)
        if not concepts:
            continue

        body = judging.body(document)
        if not body:
            session.add_all(judging.failed(document_id, [c.id for c in concepts], NO_BODY))
            processed += len(concepts)
            session.commit()
            continue

        _, omitted = judge_call(judging, document, concepts, body)
        omitted_total += omitted

        processed += len(concepts)
        if callable(progress):
            progress(processed, len(candidates))
        session.commit()

    result = stats.as_dict() | {
        "mode": judging.template.mode,
        "documents": len(by_document),
        # Concepts the model left out of its array. They count as non-matches so
        # the denominator holds, but a high number means the batch prompt is
        # overloaded and the model is dropping items.
        "omitted_by_model": omitted_total,
    }
    log.info("judge (batched): %s", result)
    return result
