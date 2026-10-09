"""Metrics sliced by concept, family, category or locus.

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

from hontology.db.lookups import concepts_by_id, get_run
from hontology.db.models import Category, Locus
from hontology.evaluation.article.verdicts import clean_verdicts
from hontology.evaluation.metrics.confusion import Confusion
from hontology.evaluation.metrics.intervals import format_ci, format_num, wilson
from hontology.evaluation.pairs import evaluate as evaluate_module
from hontology.ontology import hierarchy

# "family" is a class's top-level ancestor in the hierarchy; "category" is the
# flat grouping concepts carry independently of it.
DIMENSIONS = ("concept", "family", "category", "locus")


@dataclass
class Slice:
    key: str
    label: str
    confusion: Confusion

    def as_dict(self) -> dict:
        precision_ci = wilson(self.confusion.tp, self.confusion.tp + self.confusion.fp)
        recall_ci = wilson(self.confusion.tp, self.confusion.tp + self.confusion.fn)
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

    run = get_run(session, run_id)

    truth = evaluate_module.label_map(
        session,
        run.ontology_id,
        include_machine=include_machine,
        include_stale=include_stale,
    )

    concepts = concepts_by_id(session, run.ontology_id)
    categories = {c.id: c.name for c in session.scalars(select(Category))}
    loci = {locus.id: locus.name for locus in session.scalars(select(Locus))}
    parent_map = hierarchy.parents(session, run.ontology_id)

    def families(concept_id: int) -> list[tuple[str, str]]:
        # A class under two families counts in both; a top-level class is its own.
        tops = [a for a in hierarchy.ancestors(parent_map, concept_id) if a not in parent_map]
        return [(str(t), concepts[t].name) for t in sorted(tops or [concept_id])]

    slices: dict[str, Slice] = {}
    for verdict in clean_verdicts(session, run_id):
        key = (verdict.document_id, verdict.concept_id)
        if key not in truth:
            continue

        concept = concepts.get(verdict.concept_id)
        if concept is None:
            continue

        if dimension == "concept":
            keys = [(str(concept.id), concept.name)]
        elif dimension == "family":
            keys = families(concept.id)
        elif dimension == "category":
            keys = [
                (
                    str(concept.category_id or "none"),
                    categories.get(concept.category_id or -1, "(uncategorised)"),
                )
            ]
        else:
            keys = [
                (
                    str(verdict.locus_id or "none"),
                    loci.get(verdict.locus_id or -1, "(no locus)"),
                )
            ]

        for slice_key, label in keys:
            entry = slices.setdefault(
                slice_key, Slice(key=slice_key, label=label, confusion=Confusion())
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
        interval = format_ci(*row["precision_ci"], digits=2)
        precision, recall, f1 = (
            format_num(row[name], 6) for name in ("precision", "recall", "f1")
        )
        lines.append(
            f"{row['label'][:28]:<28} {row['n']:>4} {row['tp']:>3} {row['fp']:>3} "
            f"{row['fn']:>3}  {precision} {recall} {f1}   {interval}"
        )
    return "\n".join(lines)
