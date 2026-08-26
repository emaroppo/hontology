"""Metrics sliced by concept, category or locus.

Systematic failures only show up sliced. A pooled F1 of 0.7 can mean the judge is
uniformly mediocre, or that it is near-perfect on nine concepts and hopeless on a
tenth — and those call for completely different work. The pooled number cannot
distinguish them; this can.

Every row carries its own Wilson interval and its own count, because slicing
makes samples small fast. A concept with four labelled pairs will produce a
confident-looking 1.00 that means nothing, and the interval is what says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Category, Concept, Locus, Run, Verdict
from hontology.evalkit import evaluate as evaluate_module
from hontology.evalkit import metrics as metric_lib

DIMENSIONS = ("concept", "category", "locus")


@dataclass
class Slice:
    key: str
    label: str
    confusion: metric_lib.Confusion

    def as_dict(self) -> dict:
        precision_ci = metric_lib.wilson(
            self.confusion.tp, self.confusion.tp + self.confusion.fp
        )
        recall_ci = metric_lib.wilson(self.confusion.tp, self.confusion.tp + self.confusion.fn)
        return self.confusion.as_dict() | {
            "key": self.key,
            "label": self.label,
            "precision_ci": precision_ci,
            "recall_ci": recall_ci,
        }


def breakdown(
    session: Session,
    run_id: int,
    *,
    dimension: str = "concept",
    include_machine: bool = False,
    include_stale: bool = False,
    min_labels: int = 1,
) -> list[dict]:
    """Confusion counts and rates per slice of *dimension*.

    Slices with fewer than *min_labels* labelled pairs are still returned rather
    than hidden — a concept nobody has labelled is a finding about the label
    bank, not a row to suppress.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension {dimension!r}; expected one of {DIMENSIONS}")

    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")

    truth = evaluate_module.label_map(
        session,
        run.ontology_id,
        include_machine=include_machine,
        include_stale=include_stale,
    )

    concepts = {
        c.id: c
        for c in session.scalars(select(Concept).where(Concept.ontology_id == run.ontology_id))
    }
    categories = {c.id: c.name for c in session.scalars(select(Category))}
    loci = {locus.id: locus.name for locus in session.scalars(select(Locus))}

    slices: dict[str, Slice] = {}
    for verdict in session.scalars(
        select(Verdict).where(Verdict.run_id == run_id, Verdict.error.is_(None))
    ):
        key = (verdict.document_id, verdict.concept_id)
        if key not in truth or verdict.matched is None:
            continue

        concept = concepts.get(verdict.concept_id)
        if concept is None:
            continue

        if dimension == "concept":
            slice_key, label = str(concept.id), concept.name
        elif dimension == "category":
            slice_key = str(concept.category_id or "none")
            label = categories.get(concept.category_id or -1, "(uncategorised)")
        else:
            slice_key = str(verdict.locus_id or "none")
            label = loci.get(verdict.locus_id or -1, "(no locus)")

        entry = slices.setdefault(
            slice_key, Slice(key=slice_key, label=label, confusion=metric_lib.Confusion())
        )
        entry.confusion.add(expected=truth[key], predicted=bool(verdict.matched))

    rows = [s.as_dict() for s in slices.values() if s.confusion.total >= min_labels]
    # Worst first: the point of slicing is to find what is dragging the pool down.
    rows.sort(key=lambda r: (r["f1"] if r["f1"] is not None else 1.1, -r["n"]))
    return rows


def format_breakdown(rows: list[dict], dimension: str) -> str:
    if not rows:
        return f"(no labelled pairs to break down by {dimension})"

    lines = [
        f"{dimension:<28} {'n':>4} {'tp':>3} {'fp':>3} {'fn':>3}  "
        f"{'prec':>6} {'rec':>6} {'f1':>6}   precision 95% CI"
    ]
    lines.append("-" * 92)
    for row in rows:

        def fmt(value):
            return f"{value:>6.3f}" if value is not None else "     —"

        low, high = row["precision_ci"]
        interval = f"[{low:.2f}, {high:.2f}]" if low is not None else ""
        lines.append(
            f"{row['label'][:28]:<28} {row['n']:>4} {row['tp']:>3} {row['fp']:>3} "
            f"{row['fn']:>3}  {fmt(row['precision'])} {fmt(row['recall'])} "
            f"{fmt(row['f1'])}   {interval}"
        )
    return "\n".join(lines)
