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


def _scores(tp: float, fp: float, fn: float) -> tuple[float | None, float | None, float | None]:
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


def _totals(
    documents, counts: dict[int, _DocCounts], weights: dict[int, float] | None
) -> tuple[float, float, float]:
    tp = fp = fn = 0.0
    for doc_id in documents:
        cell = counts.get(doc_id)
        if cell is None:
            continue
        w = 1.0 if weights is None else weights.get(doc_id, 1.0)
        tp, fp, fn = tp + w * cell.tp, fp + w * cell.fp, fn + w * cell.fn
    return tp, fp, fn


def _resample(
    documents: list[int], groups: dict[int, str] | None, rng: random.Random
) -> list[int]:
    """One bootstrap draw of documents; within each group when *groups* is given.

    Resampling within groups matches a sample stratified by them: each group
    keeps its labelled count, so the weights that count on it stay fixed.
    """
    if groups is None:
        return rng.choices(documents, k=len(documents))
    by_group: dict[str, list[int]] = defaultdict(list)
    for doc_id in documents:
        by_group[groups.get(doc_id, "")].append(doc_id)
    draw: list[int] = []
    for group in sorted(by_group):
        members = by_group[group]
        draw.extend(rng.choices(members, k=len(members)))
    return draw


def document_bootstrap(
    truth: dict[Key, bool],
    predicted: dict[Key, bool],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    weights: dict[int, float] | None = None,
    groups: dict[int, str] | None = None,
) -> dict:
    """Precision, recall and F1 with 95% intervals from resampling documents.

    With *weights*, each document's counts are scaled by its weight, so a sample
    that over-represents small calendar windows still estimates the whole frame;
    *groups* makes the bootstrap resample within each window, as it was drawn.
    The tp/fp/fn reported are the unweighted counts.
    """
    counts = _per_document(truth, predicted)
    documents = sorted(counts)
    raw = _totals(documents, counts, None)
    precision, recall, f1 = _scores(*_totals(documents, counts, weights))

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot if documents else 0):
        draw = _resample(documents, groups, rng)
        scores = _scores(*_totals(draw, counts, weights))
        for name, value in zip(("precision", "recall", "f1"), scores, strict=True):
            if value is not None:
                draws[name].append(value)
    return {
        "documents": len(documents),
        "weighted": weights is not None,
        "tp": int(raw[0]),
        "fp": int(raw[1]),
        "fn": int(raw[2]),
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
    weights: dict[int, float] | None = None,
    groups: dict[int, str] | None = None,
) -> dict:
    """F1 of *arm* minus F1 of *baseline*, with a 95% interval over documents.

    Both are scored on the same resampled documents in every draw, so the
    interval reflects the difference between the systems, not the luck of which
    documents happened to be labelled. *weights* and *groups* work as in
    `document_bootstrap`.
    """
    base_counts = _per_document(truth, baseline)
    arm_counts = _per_document(truth, arm)
    documents = sorted(set(base_counts) | set(arm_counts))

    def f1_over(docs, counts) -> float | None:
        return _scores(*_totals(docs, counts, weights))[2]

    observed_base = f1_over(documents, base_counts)
    observed_arm = f1_over(documents, arm_counts)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(n_boot if documents else 0):
        draw = _resample(documents, groups, rng)
        a, b = f1_over(draw, base_counts), f1_over(draw, arm_counts)
        if a is not None and b is not None:
            differences.append(b - a)
    interval = _interval(differences)
    return {
        "documents": len(documents),
        "weighted": weights is not None,
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


def window_weights(
    manifest: dict, documents: set[int]
) -> tuple[dict[int, float], dict[int, str]]:
    """Each labelled document's weight and window, for an equal-per-window sample.

    A document's weight is its window's size in the frame over the number of
    labelled documents from that window, so pooled scores estimate the frame.
    """
    window_of = {
        row["document_id"]: row["stratum"].split("/", 1)[0] for row in manifest["order"]
    }
    sizes = manifest.get("window_sizes") or {}
    if not sizes:
        sizes = defaultdict(int)
        for window in window_of.values():
            sizes[window] += 1
    labelled: dict[str, int] = defaultdict(int)
    for doc_id in documents:
        labelled[window_of[doc_id]] += 1
    weights = {
        doc_id: sizes[window_of[doc_id]] / labelled[window_of[doc_id]] for doc_id in documents
    }
    return weights, {doc_id: window_of[doc_id] for doc_id in documents}


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
