"""Where a hierarchical arm gains and loses, on the labelled sample."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Candidate
from hontology.evaluation.article import verdicts
from hontology.ontology import hierarchy


def hierarchy_diagnostics(
    session: Session,
    run_id: int,
    baseline_id: int,
    ontology_id: int,
    truth: dict[tuple[int, int], bool],
) -> dict:
    """Where a hierarchical run gains and loses, on the labelled sample.

    Internal classes are never labelled; an internal class is positive on a
    document exactly when one of the leaves beneath it is, which is what its
    union-worded definition says. From that:

    - **recall per level**, each level conditional on a parent having been
      answered yes, so a miss is located at the level where it happened;
    - **internal false positives**, kept apart because a class answered yes
      with no labelled leaf beneath it may be a gap in the ontology rather
      than a judge error;
    - **coverage**, the run's correct leaf matches split by whether the
      baseline's retrieval had selected that pair. A gain from reaching
      leaves retrieval missed is coverage, not structure.
    """
    parent_map = hierarchy.parents(session, ontology_id)
    child_map = hierarchy.children(session, ontology_id)
    leaves = hierarchy.leaves(session, ontology_id)
    internal = set(child_map)
    documents = {doc_id for doc_id, _ in truth}

    full_truth = _with_internal_classes(truth, child_map, leaves, documents)
    answered = verdicts.answered(session, run_id, documents)
    levels = _level_counts(full_truth, answered, parent_map)

    internal_answers = [
        (key, yes) for key, yes in answered.items() if key[1] in internal and yes
    ]
    selected = _baseline_selected(session, baseline_id, documents)
    true_leaf_hits = [
        key for key, expected in truth.items() if expected and answered.get(key, False)
    ]
    return {
        "recall_by_level": {
            level: counts | {"recall": counts["hits"] / counts["n"] if counts["n"] else None}
            for level, counts in sorted(levels.items())
        },
        "internal_answered_yes": len(internal_answers),
        "internal_false_positives": sum(
            1 for key, _ in internal_answers if not full_truth[key]
        ),
        "coverage": {
            "true_positives": len(true_leaf_hits),
            "baseline_had_selected": sum(1 for key in true_leaf_hits if key in selected),
            "beyond_baseline_retrieval": sum(
                1 for key in true_leaf_hits if key not in selected
            ),
        },
    }


def _with_internal_classes(
    truth: dict[tuple[int, int], bool],
    child_map: dict[int, set[int]],
    leaves: set[int],
    documents: set[int],
) -> dict[tuple[int, int], bool]:
    """The leaf truth, plus each internal class: positive when a leaf below it is."""

    def below(class_id: int) -> set[int]:
        found: set[int] = set()
        stack = list(child_map.get(class_id, ()))
        while stack:
            current = stack.pop()
            if current not in found:
                found.add(current)
                stack.extend(child_map.get(current, ()))
        return found & leaves

    full_truth = dict(truth)
    for class_id in set(child_map):
        under = below(class_id)
        for doc_id in documents:
            full_truth[(doc_id, class_id)] = any(
                truth.get((doc_id, leaf), False) for leaf in under
            )
    return full_truth


def _level_counts(
    full_truth: dict[tuple[int, int], bool], answered: dict, parent_map: dict[int, set[int]]
) -> dict[int, dict[str, int]]:
    """Positives reached and hit at each depth, conditional on a parent's yes."""
    levels: dict[int, dict[str, int]] = {}
    for (doc_id, class_id), expected in full_truth.items():
        if not expected:
            continue
        parents = parent_map.get(class_id, set())
        if parents and not any(answered.get((doc_id, p), False) for p in parents):
            continue  # never reached: the miss belongs to a level above
        level = levels.setdefault(hierarchy.depth(parent_map, class_id), {"n": 0, "hits": 0})
        level["n"] += 1
        level["hits"] += int(answered.get((doc_id, class_id), False))
    return levels


def _baseline_selected(session: Session, baseline_id: int, documents: set[int]) -> set[Any]:
    """The pairs the baseline's retrieval selected on these documents, as rows."""
    return set(
        session.execute(
            select(Candidate.document_id, Candidate.concept_id).where(
                Candidate.run_id == baseline_id,
                Candidate.selected.is_(True),
                among(Candidate.document_id, documents),
            )
        ).all()
    )
