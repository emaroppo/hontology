"""Every lint check over one ontology, as a report."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept
from hontology.ontology.lint.duplicates import (
    DEFAULT_DUPLICATE_THRESHOLD,
    check_near_duplicates,
)
from hontology.ontology.lint.wording import check_missing_text, check_strength_drift


def lint(
    session: Session,
    ontology_id: int,
    *,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> dict:
    """Run every check. Findings are advisory, never blocking."""
    concepts = list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
    findings = (
        check_missing_text(concepts)
        + check_strength_drift(concepts)
        + check_near_duplicates(session, ontology_id, threshold=duplicate_threshold)
    )
    order = {"warning": 0, "info": 1}
    findings.sort(key=lambda f: (order.get(f.severity, 2), f.concept_name))

    return {
        "ontology_id": ontology_id,
        "concepts": len(concepts),
        "findings": [f.as_dict() for f in findings],
        "warnings": sum(1 for f in findings if f.severity == "warning"),
        "info": sum(1 for f in findings if f.severity == "info"),
    }
