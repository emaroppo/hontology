"""Hierarchical judging: each document top-down through the class hierarchy."""

from __future__ import annotations

import logging

from sqlalchemy import select

from hontology.db.models import Candidate, Concept, Document, Verdict
from hontology.ontology import hierarchy
from hontology.pipeline.judge.batched import judge_call
from hontology.pipeline.judge.context import NO_BODY, Judging

log = logging.getLogger(__name__)


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
        answered = _answered(judging, document_id)
        body = judging.body(document)

        while sets := _next_sets(top, children, answered):
            for class_ids in sets:
                concepts = judging.concepts(class_ids)
                if not body:
                    ids = [concept.id for concept in concepts]
                    session.add_all(judging.failed(document_id, ids, NO_BODY))
                    answered.update(dict.fromkeys(ids))
                    session.commit()
                    continue
                for group, routing in _groups(judging, concepts, children):
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


def _answered(judging: Judging, document_id: int) -> dict[int, bool | None]:
    """What is already stored for *document_id*, so a restart resumes mid-document."""
    return {
        concept_id: (None if error else matched)
        for concept_id, matched, error in judging.session.execute(
            select(Verdict.concept_id, Verdict.matched, Verdict.error).where(
                Verdict.run_id == judging.run_id, Verdict.document_id == document_id
            )
        )
    }


def _next_sets(
    top: list[int], children: dict[int, set[int]], answered: dict[int, bool | None]
) -> list[list[int]]:
    """The sibling sets to ask next: unasked top classes, and the unasked
    children of every class answered yes, each class in one set at most."""
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
    return sets


def _groups(
    judging: Judging, concepts: list[Concept], children: dict[int, set[int]]
) -> list[tuple[list[Concept], bool]]:
    """One sibling set's calls, as ``(concepts, asked as routing questions)``.

    With routing questions, parents and leaves of one sibling set are asked in
    separate calls, so leaves see exactly the prompt the flat arm uses.
    """
    if judging.template.build_route is None:
        return [(concepts, False)]
    groups = [
        ([c for c in concepts if children.get(c.id)], True),
        ([c for c in concepts if not children.get(c.id)], False),
    ]
    return [(group, routing) for group, routing in groups if group]
