"""Article-level scoring on the labelled document sample.

Two views of a run, both over documents a person labelled against every concept:

- **End to end:** a pair the run never judged counts as *no*. This is what the
  system as a whole delivers, retrieval and any budget included.
- **Judge only:** just the pairs the run judged, which isolates the judge.

**Why the bootstrap resamples documents.** One article carries a label for every
concept, and those labels are not independent: an article about a typhoon is
negative on forty-odd concepts for the same reason. Resampling single pairs
would treat them as independent and give intervals that are too narrow, so every
interval here resamples whole documents. The difference between two runs is
*paired*: both are scored on the same resampled documents each time.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Verdict
from hontology.evalkit.metrics import Confusion

Key = tuple[int, int]
DEFAULT_BOOTSTRAP = 2000


def predictions(session: Session, run_id: int, keys: set[Key]) -> dict[Key, bool]:
    """The run's answer on each labelled pair; unjudged and errored pairs are no."""
    documents = {doc_id for doc_id, _ in keys}
    judged = {
        (doc_id, concept_id): bool(matched)
        for doc_id, concept_id, matched in session.execute(
            select(Verdict.document_id, Verdict.concept_id, Verdict.matched).where(
                Verdict.run_id == run_id,
                among(Verdict.document_id, documents),
                Verdict.error.is_(None),
                Verdict.matched.is_not(None),
            )
        )
    }
    return {key: judged.get(key, False) for key in keys}


def judged_keys(session: Session, run_id: int, keys: set[Key]) -> set[Key]:
    documents = {doc_id for doc_id, _ in keys}
    found = {
        (doc_id, concept_id)
        for doc_id, concept_id in session.execute(
            select(Verdict.document_id, Verdict.concept_id).where(
                Verdict.run_id == run_id,
                among(Verdict.document_id, documents),
                Verdict.error.is_(None),
                Verdict.matched.is_not(None),
            )
        )
    }
    return keys & found


def confusion(truth: dict[Key, bool], predicted: dict[Key, bool], keys=None) -> Confusion:
    result = Confusion()
    for key in keys if keys is not None else truth:
        result.add(expected=truth[key], predicted=predicted.get(key, False))
    return result


@dataclass
class _DocCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _per_document(truth: dict[Key, bool], predicted: dict[Key, bool]) -> dict[int, _DocCounts]:
    counts: dict[int, _DocCounts] = defaultdict(_DocCounts)
    for (doc_id, concept_id), expected in truth.items():
        guess = predicted.get((doc_id, concept_id), False)
        cell = counts[doc_id]
        if expected and guess:
            cell.tp += 1
        elif guess:
            cell.fp += 1
        elif expected:
            cell.fn += 1
    return counts


def _scores(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return precision, recall, f1


def _interval(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return [round(low, 4), round(high, 4)]


def document_bootstrap(
    truth: dict[Key, bool],
    predicted: dict[Key, bool],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Precision, recall and F1 with 95% intervals from resampling documents."""
    counts = _per_document(truth, predicted)
    documents = sorted(counts)
    total = _DocCounts()
    for cell in counts.values():
        total.tp, total.fp, total.fn = (
            total.tp + cell.tp,
            total.fp + cell.fp,
            total.fn + cell.fn,
        )
    precision, recall, f1 = _scores(total.tp, total.fp, total.fn)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot if documents else 0):
        tp = fp = fn = 0
        for doc_id in rng.choices(documents, k=len(documents)):
            cell = counts[doc_id]
            tp, fp, fn = tp + cell.tp, fp + cell.fp, fn + cell.fn
        for name, value in zip(("precision", "recall", "f1"), _scores(tp, fp, fn), strict=True):
            if value is not None:
                draws[name].append(value)
    return {
        "documents": len(documents),
        "tp": total.tp,
        "fp": total.fp,
        "fn": total.fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_ci": _interval(draws["precision"]),
        "recall_ci": _interval(draws["recall"]),
        "f1_ci": _interval(draws["f1"]),
    }


def paired_document_bootstrap(
    truth: dict[Key, bool],
    baseline: dict[Key, bool],
    arm: dict[Key, bool],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """F1 of *arm* minus F1 of *baseline*, with a 95% interval over documents.

    Both are scored on the same resampled documents in every draw, so the
    interval reflects the difference between the systems, not the luck of which
    documents happened to be labelled.
    """
    base_counts = _per_document(truth, baseline)
    arm_counts = _per_document(truth, arm)
    documents = sorted(set(base_counts) | set(arm_counts))

    def f1_over(docs, counts) -> float | None:
        tp = sum(counts[d].tp for d in docs if d in counts)
        fp = sum(counts[d].fp for d in docs if d in counts)
        fn = sum(counts[d].fn for d in docs if d in counts)
        return _scores(tp, fp, fn)[2]

    observed_base = f1_over(documents, base_counts)
    observed_arm = f1_over(documents, arm_counts)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(n_boot if documents else 0):
        draw = rng.choices(documents, k=len(documents))
        a, b = f1_over(draw, base_counts), f1_over(draw, arm_counts)
        if a is not None and b is not None:
            differences.append(b - a)
    interval = _interval(differences)
    return {
        "documents": len(documents),
        "baseline_f1": observed_base,
        "arm_f1": observed_arm,
        "difference": (
            observed_arm - observed_base
            if observed_arm is not None and observed_base is not None
            else None
        ),
        "difference_ci": interval,
        # The pre-registered decision rule: an improvement only if the interval
        # lies entirely above zero.
        "improvement": interval[0] is not None and interval[0] > 0,
    }


def half_width(ci: list[float | None]) -> float | None:
    low, high = ci
    return None if low is None or high is None else round((high - low) / 2, 4)


def sample_status(scores: dict, *, target: float = 0.05) -> dict:
    """Whether the baseline's intervals are narrow enough to stop labelling.

    Looks only at the baseline's own precision and recall, never at a
    difference between arms, so stopping cannot be timed to favour an arm.
    """
    widths = {
        "precision": half_width(scores["precision_ci"]),
        "recall": half_width(scores["recall_ci"]),
    }
    met = all(width is not None and width <= target for width in widths.values())
    return {
        "documents": scores["documents"],
        "positives": scores["tp"] + scores["fn"],
        "half_widths": widths,
        "target": target,
        "target_met": met,
    }
