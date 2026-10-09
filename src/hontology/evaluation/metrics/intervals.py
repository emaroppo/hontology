"""Intervals on the headline metrics, and how they are printed.

Precision and recall get a Wilson interval; F1, which is not a simple
proportion, a seeded bootstrap over the labelled pairs.
"""

from __future__ import annotations

import math
import random

from hontology.evaluation.metrics.confusion import Confusion


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
