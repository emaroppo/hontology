"""Metrics, with the uncertainty attached.

A point estimate from a few hundred labels invites over-reading. A ten-point F1
gap on thirty pairs is noise, and reporting it as a finding is how a pipeline
acquires "improvements" that do not survive contact with more data. So every
headline number carries an interval, and comparing two runs uses a *paired* test
over the pairs they both judged.

Intervals are derived from confusion counts rather than stored separately, so
they stay correct when counts are aggregated across runs or slices:

- **precision and recall** are proportions, so they get a **Wilson** score
  interval — closed form, and well behaved near 0 and 1 and at small n where the
  normal approximation produces bounds outside [0, 1].
- **F1** is not a simple proportion, so its interval comes from a **bootstrap**
  over the labelled pairs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else None

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.total if self.total else None

    def add(self, *, expected: bool, predicted: bool) -> None:
        if expected and predicted:
            self.tp += 1
        elif expected and not predicted:
            self.fn += 1
        elif not expected and predicted:
            self.fp += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a proportion (95% by default).

    Preferred over the normal approximation because it stays inside [0, 1] and
    remains sensible at small n — exactly the regime a hand-built label bank
    lives in.
    """
    if n <= 0:
        return (None, None)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_f1(
    confusion: Confusion,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for F1.

    Resamples the labelled pairs with replacement from their observed category
    proportions and recomputes F1 each time. Seeded, so a reported interval is
    reproducible.
    """
    total = confusion.total
    if total == 0 or (2 * confusion.tp + confusion.fp + confusion.fn) == 0:
        return (None, None)

    rng = random.Random(seed)
    weights = [confusion.tp, confusion.fp, confusion.fn, confusion.tn]
    samples: list[float] = []
    for _ in range(n_boot):
        draw = rng.choices((0, 1, 2, 3), weights=weights, k=total)
        tp = draw.count(0)
        fp = draw.count(1)
        fn = draw.count(2)
        denominator = 2 * tp + fp + fn
        samples.append(2 * tp / denominator if denominator else 0.0)

    samples.sort()
    low_q = (1 - ci) / 2
    return (
        samples[int(low_q * n_boot)],
        samples[min(n_boot - 1, int((1 - low_q) * n_boot))],
    )


def format_ci(low: float | None, high: float | None, *, digits: int = 3) -> str:
    if low is None or high is None:
        return ""
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def format_num(value: float | None, width: int = 0, *, digits: int = 3) -> str:
    """*value* to *digits* places, right-aligned in *width*; a dash when absent."""
    if value is None:
        return " " * (width - 1) + "—"
    return f"{value:>{width}.{digits}f}" if width else f"{value:.{digits}f}"


def with_intervals(confusion: Confusion, *, seed: int = 0) -> dict:
    """Confusion counts, rates, and an interval for each headline metric."""
    precision_ci = wilson(confusion.tp, confusion.tp + confusion.fp)
    recall_ci = wilson(confusion.tp, confusion.tp + confusion.fn)
    f1_ci = bootstrap_f1(confusion, seed=seed)
    return confusion.as_dict() | {
        "precision_ci": precision_ci,
        "recall_ci": recall_ci,
        "f1_ci": f1_ci,
    }


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


@dataclass
class McNemarResult:
    """Paired comparison over items both runs judged.

    ``b`` counts items the first run got right and the second wrong; ``c`` the
    reverse. Items both got right or both got wrong carry no information about
    which is better, which is precisely why an unpaired test wastes them.
    """

    both_correct: int = 0
    only_a_correct: int = 0  # b
    only_b_correct: int = 0  # c
    both_wrong: int = 0
    n_pairs: int = 0

    @property
    def discordant_count(self) -> int:
        """Pairs the two runs answered differently — the only informative ones."""
        return self.only_a_correct + self.only_b_correct

    @property
    def statistic(self) -> float | None:
        b, c = self.only_a_correct, self.only_b_correct
        if b + c == 0:
            return None
        # Continuity-corrected, appropriate for the small discordant counts a
        # hand-built label bank produces.
        return (abs(b - c) - 1) ** 2 / (b + c)

    @property
    def p_value(self) -> float | None:
        statistic = self.statistic
        if statistic is None:
            return None
        # Survival function of chi-square with one degree of freedom.
        return math.erfc(math.sqrt(max(0.0, statistic) / 2))

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "both_correct": self.both_correct,
            "only_a_correct": self.only_a_correct,
            "only_b_correct": self.only_b_correct,
            "both_wrong": self.both_wrong,
            "discordant": self.discordant_count,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }


def mcnemar(
    truth: dict[tuple[int, int], bool],
    a: dict[tuple[int, int], bool],
    b: dict[tuple[int, int], bool],
) -> McNemarResult:
    """Compare two runs on the pairs they both judged and that carry a label."""
    result = McNemarResult()
    for key, expected in truth.items():
        if key not in a or key not in b:
            continue
        result.n_pairs += 1
        a_right = a[key] == expected
        b_right = b[key] == expected
        if a_right and b_right:
            result.both_correct += 1
        elif a_right and not b_right:
            result.only_a_correct += 1
        elif b_right and not a_right:
            result.only_b_correct += 1
        else:
            result.both_wrong += 1
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Scored over the pre-cutoff pool, so ranking is separable from the cutoff.

    ``precision`` is computed only over candidates that carry a label, and
    ``coverage`` reports that denominator: an unlabelled candidate is *unknown*,
    never silently counted as a false positive. Without the coverage number a
    precision figure over a sparsely labelled corpus is unreadable.
    """

    recall_at_k: dict[int, float | None] = field(default_factory=dict)
    mrr: float | None = None
    n_positives: int = 0
    n_positives_ranked: int = 0
    labelled_candidates: int = 0
    total_candidates: int = 0
    true_positives: int = 0
    false_positives: int = 0
    selected_recall_num: int = 0
    selected_recall_den: int = 0

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def coverage(self) -> float | None:
        return (
            self.labelled_candidates / self.total_candidates if self.total_candidates else None
        )

    @property
    def cutoff_recall(self) -> float | None:
        """Of the positives retrieval could have kept, how many survived the cutoff."""
        return (
            self.selected_recall_num / self.selected_recall_den
            if self.selected_recall_den
            else None
        )

    def as_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "n_positives": self.n_positives,
            "n_positives_ranked": self.n_positives_ranked,
            "precision": self.precision,
            "coverage": self.coverage,
            "labelled_candidates": self.labelled_candidates,
            "total_candidates": self.total_candidates,
            "cutoff_recall": self.cutoff_recall,
        }


def retrieval_metrics(
    labels: dict[tuple[int, int], bool],
    pool: dict[tuple[int, int], int],
    selected: set[tuple[int, int]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Score candidate selection against pair labels.

    ``pool`` maps a pair to its 1-based rank before the cutoff; ``selected`` is
    the subset that survived it.
    """
    metrics = RetrievalMetrics(total_candidates=len(pool))

    for key in pool:
        if key in labels:
            metrics.labelled_candidates += 1
            if labels[key]:
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1

    positives = [key for key, expected in labels.items() if expected]
    metrics.n_positives = len(positives)
    ranked_positives = [key for key in positives if key in pool]
    metrics.n_positives_ranked = len(ranked_positives)

    for k in ks:
        metrics.recall_at_k[k] = (
            sum(1 for key in ranked_positives if pool[key] <= k) / len(positives)
            if positives
            else None
        )

    if ranked_positives:
        metrics.mrr = sum(1 / pool[key] for key in ranked_positives) / len(positives)

    metrics.selected_recall_den = len(ranked_positives)
    metrics.selected_recall_num = sum(1 for key in ranked_positives if key in selected)
    return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_bins(pairs: list[tuple[float, bool]], *, n_bins: int = 5) -> list[dict]:
    """Observed accuracy per confidence bin.

    A well-calibrated model is right about 70% of the time when it says 0.7. This
    is what shows, concretely, whether a confidence number means anything.
    """
    bins: list[dict] = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        members = [
            correct
            for confidence, correct in pairs
            if (low <= confidence < high) or (index == n_bins - 1 and confidence == 1.0)
        ]
        bins.append(
            {
                "range": (round(low, 3), round(high, 3)),
                "n": len(members),
                "accuracy": (sum(members) / len(members)) if members else None,
            }
        )
    return bins
