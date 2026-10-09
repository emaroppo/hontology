"""The labelled sample's design: window weights, and when to stop labelling."""

from __future__ import annotations

from collections import defaultdict


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
