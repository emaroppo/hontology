"""The rows a run's findings export as: one per detection, one per event, each
with its verification status against the ground-truth bank."""

from __future__ import annotations

from dataclasses import dataclass, field

DETECTION_COLUMNS = [
    "run_id",
    "concept",
    "locus_iso3",
    "locus_iso2",
    "occurred_on",
    "document_url",
    "document_title",
    "confidence",
    "vote_fraction",
    "evidence",
    "verification",
    "model",
    "prompt_id",
    "ontology_version",
]

EVENT_COLUMNS = [
    "run_id",
    "concept",
    "locus_iso3",
    "occurred_on",
    "documents",
    "confirmed_documents",
    "rejected_documents",
    "max_confidence",
    "verification",
    "sample_url",
]

# How a detection stands against the ground-truth bank.
CONFIRMED = "confirmed"
REJECTED = "rejected"
UNVERIFIED = "unverified"


@dataclass
class Detection:
    run_id: int
    concept: str
    locus_iso3: str | None
    locus_iso2: str | None
    occurred_on: str | None
    document_id: int
    document_url: str
    document_title: str | None
    confidence: float | None
    vote_fraction: float | None
    evidence: str | None
    verification: str
    model: str | None
    prompt_id: str | None
    ontology_version: str

    def as_row(self) -> dict:
        values = ((column, getattr(self, column)) for column in DETECTION_COLUMNS)
        return {column: "" if value is None else value for column, value in values}


@dataclass
class Event:
    """One occurrence, however many documents reported it."""

    run_id: int
    concept: str
    locus_iso3: str | None
    occurred_on: str | None
    document_ids: set[int] = field(default_factory=set)
    confirmed: int = 0
    rejected: int = 0
    max_confidence: float | None = None
    sample_url: str = ""

    @property
    def verification(self) -> str:
        """An event is confirmed if any supporting document was confirmed.

        Rejected only when every supporting document was rejected — one bad
        article does not disprove an event the others evidence.
        """
        if self.confirmed:
            return CONFIRMED
        if self.rejected and self.rejected == len(self.document_ids):
            return REJECTED
        return UNVERIFIED

    def as_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "concept": self.concept,
            "locus_iso3": self.locus_iso3 or "",
            "occurred_on": self.occurred_on or "",
            "documents": len(self.document_ids),
            "confirmed_documents": self.confirmed,
            "rejected_documents": self.rejected,
            "max_confidence": self.max_confidence if self.max_confidence is not None else "",
            "verification": self.verification,
            "sample_url": self.sample_url,
        }
