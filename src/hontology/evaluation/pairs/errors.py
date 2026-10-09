"""Error triage: every mistake, with the model's own reasoning attached.

A confusion matrix says *how many* went wrong. It never says *why*, and the why
is where the fixable patterns are — a whole class of false positives turning out
to be planned-but-not-yet-happened events, or an exclusion criterion the model is
quietly ignoring.

The judge already records its evidence quote and, when thinking is enabled, its
reasoning trace. Both were being stored and never read. This joins them to the
misclassifications so failures can be read in bulk instead of grepped one at a
time.

False positives and false negatives are separated because they have different
cures: a false positive usually means the concept's exclusions are too weak,
while a false negative usually means retrieval or the definition is too narrow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import concept_names, get_run
from hontology.db.models import Document, PairLabel
from hontology.evaluation.article.verdicts import clean_verdicts
from hontology.evaluation.metrics.intervals import format_num
from hontology.evaluation.pairs import evaluate as evaluate_module


@dataclass
class ErrorRow:
    kind: str  # "false_positive" | "false_negative"
    document_id: int
    concept_id: int
    concept_name: str
    document_url: str
    document_title: str | None
    expected: bool
    predicted: bool
    confidence: float | None
    evidence: str | None
    reasoning: str | None
    label_note: str | None
    label_source: str

    def as_dict(self) -> dict:
        return asdict(self)


def triage(
    session: Session,
    run_id: int,
    *,
    kind: str | None = None,
    include_machine: bool = False,
    include_stale: bool = False,
    limit: int = 100,
) -> list[ErrorRow]:
    """Every misclassified pair for a run, with its evidence and reasoning.

    Sorted most-confident-first: a mistake the model was sure about is more
    diagnostic than one it nearly got right, because it points at a systematic
    misreading rather than a borderline call.
    """
    run = get_run(session, run_id)

    truth = evaluate_module.label_map(
        session,
        run.ontology_id,
        include_machine=include_machine,
        include_stale=include_stale,
    )
    concepts = concept_names(session, run.ontology_id)
    labels = {
        (label.document_id, label.concept_id): label
        for label in session.scalars(select(PairLabel))
    }

    rows: list[ErrorRow] = []
    for verdict in clean_verdicts(session, run_id):
        key = (verdict.document_id, verdict.concept_id)
        if key not in truth:
            continue

        expected = truth[key]
        predicted = bool(verdict.matched)
        if expected == predicted:
            continue

        row_kind = "false_positive" if predicted else "false_negative"
        if kind and row_kind != kind:
            continue

        document = session.get(Document, verdict.document_id)
        label = labels.get(key)
        rows.append(
            ErrorRow(
                kind=row_kind,
                document_id=verdict.document_id,
                concept_id=verdict.concept_id,
                concept_name=concepts.get(verdict.concept_id, "?"),
                document_url=document.url if document else "",
                document_title=document.title if document else None,
                expected=expected,
                predicted=predicted,
                confidence=verdict.confidence,
                evidence=verdict.evidence,
                reasoning=verdict.reasoning,
                label_note=label.note if label else None,
                label_source=label.source if label else "?",
            )
        )

    rows.sort(key=lambda r: r.confidence or 0.0, reverse=True)
    return rows[:limit]


def summary(session: Session, run_id: int, **kwargs) -> dict:
    """Counts by kind and by concept — where the errors concentrate."""
    rows = triage(session, run_id, limit=10_000, **kwargs)
    by_concept: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = by_concept.setdefault(
            row.concept_name, {"false_positive": 0, "false_negative": 0}
        )
        entry[row.kind] += 1

    return {
        "total": len(rows),
        "false_positives": sum(1 for r in rows if r.kind == "false_positive"),
        "false_negatives": sum(1 for r in rows if r.kind == "false_negative"),
        "by_concept": dict(
            sorted(
                by_concept.items(),
                key=lambda kv: -(kv[1]["false_positive"] + kv[1]["false_negative"]),
            )
        ),
        "with_evidence": sum(1 for r in rows if r.evidence),
        "with_reasoning": sum(1 for r in rows if r.reasoning),
    }


def format_triage(rows: list[ErrorRow], *, max_chars: int = 200) -> str:
    if not rows:
        return "(no misclassified pairs — either the run is perfect or nothing is labelled)"

    lines: list[str] = []
    for row in rows:
        marker = "FP" if row.kind == "false_positive" else "FN"
        confidence = format_num(row.confidence, digits=2)
        lines.append(
            f"[{marker}] {row.concept_name}  conf={confidence}  "
            f"{(row.document_title or row.document_url)[:60]}"
        )
        if row.evidence:
            lines.append(f'      model quoted: "{row.evidence[:max_chars]}"')
        if row.label_note:
            lines.append(f"      label note:   {row.label_note[:max_chars]}")
        if row.reasoning:
            lines.append(f"      reasoning:    {row.reasoning[:max_chars]}…")
        lines.append("")
    return "\n".join(lines)
