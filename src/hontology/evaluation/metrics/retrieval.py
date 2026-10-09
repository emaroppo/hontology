"""Retrieval metrics over the pre-cutoff candidate pool."""

from __future__ import annotations

from dataclasses import dataclass, field


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
