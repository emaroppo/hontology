"""Judging several concepts per call: one call per document, or top-down."""

from __future__ import annotations

import json
import logging

from sqlalchemy import select

from hontology.db.models import Candidate, Concept, Document, Verdict
from hontology.judge.context import NO_BODY, Judging, share
from hontology.judge.parse import parse_batch
from hontology.judge.providers.base import ProviderError
from hontology.ontology import hierarchy

log = logging.getLogger(__name__)


def judge_call(
    judging: Judging,
    document: Document,
    concepts: list[Concept],
    body: str,
    *,
    routing: bool = False,
) -> tuple[dict[int, bool | None], int]:
    """One batched call over *concepts*, writing a verdict for each.

    Returns each concept's answer (None when the call failed) and how many
    concepts the model left out of its reply. Shared by batched and
    hierarchical judging, so both ask the judge in exactly the same way.
    With *routing*, the concepts are parent classes asked as routing questions.
    """
    template, session = judging.template, judging.session
    if routing:
        assert template.build_route is not None and template.route_system is not None
        prompt = template.build_route(document, concepts, body, judging.body_limit)
        system = template.route_system
    else:
        assert template.build_batch is not None  # guaranteed by PromptTemplate
        prompt = template.build_batch(document, concepts, body, judging.body_limit)
        system = template.system

    try:
        completion = judging.ask(system, prompt)
        parsed = parse_batch(completion.text, [c.id for c in concepts])
        judging.stats.spent(completion)
    except (ProviderError, json.JSONDecodeError) as exc:
        # One bad response costs the whole set — recorded per pair so the
        # denominator stays honest and liveness catches it.
        message = f"{type(exc).__name__}: {exc}"
        session.add_all(judging.failed(document.id, [c.id for c in concepts], message))
        return {concept.id: None for concept in concepts}, 0

    omitted = 0
    answers: dict[int, bool | None] = {}
    for position, concept in enumerate(concepts):
        result = parsed[concept.id]
        if result.get("omitted"):
            omitted += 1
        session.add(
            judging.verdict(
                document.id,
                concept.id,
                matched=result["matched"],
                confidence=result["confidence"],
                vote_fraction=1.0,
                locus_id=judging.iso2_to_locus.get(result["country"]),
                evidence=result["evidence"] or None,
                reasoning=completion.reasoning or None,
                # One call served the whole set, so attributing its full cost
                # to each pair would multiply the real cost.
                latency_s=completion.latency_s / len(concepts),
                input_tokens=share(completion.input_tokens, len(concepts), position),
                output_tokens=share(completion.output_tokens, len(concepts), position),
            )
        )
        answers[concept.id] = bool(result["matched"])
        judging.stats.judged += 1
        judging.stats.matched += int(result["matched"])
    return answers, omitted


def judge_hierarchical(
    judging: Judging,
    *,
    ontology_id: int,
    candidates: list[Candidate],
    limit: int | None,
    progress: object | None,
) -> dict:
    """Judge each document top-down through the class hierarchy.

    A document is judged if retrieval selected at least one leaf for it, the
    same gate the flat arm passes through. The top-level classes are asked in
    one call; then, round by round, the not-yet-answered children of every class
    answered yes are asked, one call per sibling set, until nothing new opens.

    - **Each class at most once per document.** A class with two parents
      answered yes is asked under the first only: a second parent means
      *either*, so one answer serves both.
    - **No descent below a failure.** An errored call counts as not positive,
      so the branch beneath it is left unasked rather than guessed at.
    - **Leaves under a "no" get no row.** Metrics read them as negatives.
    - **Resumable.** The descent state is rebuilt from the verdicts already
      stored, so a restart continues mid-document without re-asking anything.
    """
    session, template, stats = judging.session, judging.template, judging.stats
    children = hierarchy.children(session, ontology_id)
    top = sorted(hierarchy.top_level(session, ontology_id))
    documents = sorted({c.document_id for c in candidates})
    stats.total = len(documents)
    processed = 0
    calls = 0
    omitted_total = 0

    for done_documents, document_id in enumerate(documents, start=1):
        if limit is not None and processed >= limit:
            break
        document = session.get(Document, document_id)
        if document is None:
            continue
        answered: dict[int, bool | None] = {
            concept_id: (None if error else matched)
            for concept_id, matched, error in session.execute(
                select(Verdict.concept_id, Verdict.matched, Verdict.error).where(
                    Verdict.run_id == judging.run_id, Verdict.document_id == document_id
                )
            )
        }
        body = judging.body(document)

        while True:
            sets: list[list[int]] = []
            queued: set[int] = set()
            pending_top = [c for c in top if c not in answered]
            if pending_top:
                sets.append(pending_top)
                queued.update(pending_top)
            for parent in sorted(c for c, yes in answered.items() if yes):
                below = sorted(
                    c for c in children.get(parent, ()) if c not in answered and c not in queued
                )
                if below:
                    sets.append(below)
                    queued.update(below)
            if not sets:
                break

            for class_ids in sets:
                concepts = judging.concepts(class_ids)
                if not body:
                    ids = [concept.id for concept in concepts]
                    session.add_all(judging.failed(document_id, ids, NO_BODY))
                    answered.update(dict.fromkeys(ids))
                    session.commit()
                    continue
                # With routing questions, parents and leaves of one sibling set
                # are asked in separate calls, so leaves see exactly the prompt
                # the flat arm uses.
                groups = (
                    [
                        ([c for c in concepts if children.get(c.id)], True),
                        ([c for c in concepts if not children.get(c.id)], False),
                    ]
                    if template.build_route is not None
                    else [(concepts, False)]
                )
                for group, routing in groups:
                    if not group:
                        continue
                    answers, omitted = judge_call(
                        judging, document, group, body, routing=routing
                    )
                    answered.update(answers)
                    omitted_total += omitted
                    processed += len(group)
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
