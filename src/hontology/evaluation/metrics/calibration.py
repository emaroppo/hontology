"""Calibration: observed accuracy per confidence bin."""

from __future__ import annotations


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
