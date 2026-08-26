"""Is each code earning its place in the filter?

The pre-scrape filter admits a document when one of its feed codes maps to a
concept. That gate bounds recall for everything downstream — an article it turns
away can never be detected, no matter how good retrieval and the judge are — so
the mapping deserves evidence rather than intuition.

For each code this reports what it admitted and what those admissions were worth:
documents let in, how many were labelled, and how many of those labels turned out
positive. A code admitting hundreds of documents that never yield a true match is
costing scrape budget and buying nothing; a code admitting few but almost all
true is cheap and precise.

The counts are deliberately reported next to each other rather than reduced to a
single yield number. A code with two labelled admissions out of eighty is not
"2.5% useful" — it is *unmeasured*, and a ratio would hide that.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Code, Concept, Document, PairLabel
from hontology.ingest import filter as ingest_filter


@dataclass
class CodeReport:
    code: str
    level: str
    name: str | None
    concepts: int = 0
    documents_admitted: int = 0
    documents_fetched: int = 0
    documents_junk: int = 0
    labelled_pairs: int = 0
    positive_pairs: int = 0
    concept_ids: set[int] = field(default_factory=set)

    @property
    def yield_rate(self) -> float | None:
        """Positive share of *labelled* admissions, or None when unmeasured."""
        return self.positive_pairs / self.labelled_pairs if self.labelled_pairs else None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "level": self.level,
            "name": self.name,
            "concepts": self.concepts,
            "documents_admitted": self.documents_admitted,
            "documents_fetched": self.documents_fetched,
            "documents_junk": self.documents_junk,
            "labelled_pairs": self.labelled_pairs,
            "positive_pairs": self.positive_pairs,
            "yield_rate": self.yield_rate,
        }


def report(session: Session, ontology_id: int) -> dict:
    """Per-code cost and benefit for one ontology's filter mapping."""
    links = ingest_filter.concept_code_map(session, ontology_id)
    if not links:
        return {
            "usable": False,
            "reason": (
                "this ontology has no concept↔code links, so there is no filter to report on"
            ),
            "codes": [],
        }

    code_rows = {
        c.code: c for c in session.scalars(select(Code).where(Code.code.in_(list(links))))
    }
    concept_ids = {
        c.id for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }

    labels_by_document: dict[int, list[PairLabel]] = defaultdict(list)
    for label in session.scalars(
        select(PairLabel).where(PairLabel.concept_id.in_(concept_ids))
    ):
        labels_by_document[label.document_id].append(label)

    documents = {d.id: d for d in session.scalars(select(Document))}
    matches = ingest_filter.matching_documents(session, ontology_id)

    reports: dict[str, CodeReport] = {}
    for match in matches.values():
        code_row = code_rows.get(match.code)
        entry = reports.setdefault(
            match.code,
            CodeReport(
                code=match.code,
                level=match.level,
                name=code_row.name if code_row else None,
            ),
        )
        entry.concept_ids |= set(match.concept_ids)
        entry.documents_admitted += 1

        document = documents.get(match.document_id)
        if document is not None:
            if document.fetched_at is not None:
                entry.documents_fetched += 1
            if document.is_junk:
                entry.documents_junk += 1

        for label in labels_by_document.get(match.document_id, []):
            # Only labels for concepts this code actually reaches.
            if label.concept_id not in match.concept_ids:
                continue
            entry.labelled_pairs += 1
            if label.matched:
                entry.positive_pairs += 1

    for entry in reports.values():
        entry.concepts = len(entry.concept_ids)

    rows = sorted(
        (r.as_dict() for r in reports.values()),
        key=lambda r: (-(r["documents_admitted"]), r["code"]),
    )
    admitted = sum(r["documents_admitted"] for r in rows)
    labelled = sum(r["labelled_pairs"] for r in rows)

    return {
        "usable": True,
        "codes": rows,
        "totals": {
            "codes": len(rows),
            "documents_admitted": admitted,
            "labelled_pairs": labelled,
            "positive_pairs": sum(r["positive_pairs"] for r in rows),
            "unmeasured_codes": sum(1 for r in rows if r["labelled_pairs"] == 0),
        },
    }


def format_report(result: dict) -> str:
    if not result.get("usable"):
        return result.get("reason", "(nothing to report)")
    if not result["codes"]:
        return "(no documents admitted by any code yet)"

    lines = [
        f"{'code':<8} {'level':<6} {'admitted':>9} {'fetched':>8} {'junk':>5} "
        f"{'labelled':>9} {'positive':>9} {'yield':>7}  name"
    ]
    lines.append("-" * 100)
    for row in result["codes"]:
        rate = row["yield_rate"]
        shown = f"{rate:>6.1%}" if rate is not None else "     —"
        lines.append(
            f"{row['code']:<8} {row['level']:<6} {row['documents_admitted']:>9} "
            f"{row['documents_fetched']:>8} {row['documents_junk']:>5} "
            f"{row['labelled_pairs']:>9} {row['positive_pairs']:>9} {shown}  "
            f"{(row['name'] or '')[:34]}"
        )

    totals = result["totals"]
    lines.append("")
    lines.append(
        f"{totals['codes']} code(s) admitted {totals['documents_admitted']} document(s); "
        f"{totals['labelled_pairs']} labelled pair(s), {totals['positive_pairs']} positive."
    )
    if totals["unmeasured_codes"]:
        lines.append(
            f"{totals['unmeasured_codes']} code(s) have no labelled admissions yet — "
            "unmeasured, not worthless. Label some of what they admit before cutting them."
        )
    return "\n".join(lines)
