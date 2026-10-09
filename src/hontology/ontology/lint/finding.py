"""One lint finding: a check, the concept it is about, and why."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    check: str
    severity: str
    concept_id: int
    concept_name: str
    message: str
    related_concept_id: int | None = None
    score: float | None = None

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "message": self.message,
            "related_concept_id": self.related_concept_id,
            "score": self.score,
        }
