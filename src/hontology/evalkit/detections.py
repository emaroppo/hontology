"""Getting the findings out.

Everything else in this package exports the machinery's *inputs* (an ontology) or
its *scores* (metrics, a leaderboard). This exports its output: the things the
pipeline actually found. Without it the product's result is reachable only by
querying Postgres directly, which makes the whole system a closed loop.

Two shapes, because two different consumers want different things:

- **detections** — one row per matched `(document, concept)` pair, with the
  evidence quote. This is the audit trail: every claim traceable to the article
  and sentence it came from.
- **events** — one row per `(concept, locus, date)`, aggregating the documents
  that support it. This is what a downstream model wants, because "a riot
  happened in Kenya on the 3rd" is one fact regardless of how many outlets
  carried it.

**Every row carries its verification status.** A detection is a model's claim,
not a fact, and a consumer must be able to tell a confirmed one from an
unreviewed one. Exporting them indistinguishably would launder model output into
apparent ground truth — the same confusion the label provenance rules exist to
prevent, just one stage further out.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    TRUSTED_SOURCES,
    Concept,
    Document,
    FeedEvent,
    Locus,
    PairLabel,
    Run,
    Verdict,
)

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
        return {
            "run_id": self.run_id,
            "concept": self.concept,
            "locus_iso3": self.locus_iso3 or "",
            "locus_iso2": self.locus_iso2 or "",
            "occurred_on": self.occurred_on or "",
            "document_url": self.document_url,
            "document_title": self.document_title or "",
            "confidence": self.confidence if self.confidence is not None else "",
            "vote_fraction": self.vote_fraction if self.vote_fraction is not None else "",
            "evidence": self.evidence or "",
            "verification": self.verification,
            "model": self.model or "",
            "prompt_id": self.prompt_id or "",
            "ontology_version": self.ontology_version,
        }


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


def _document_dates(session: Session, document_ids: list[int]) -> dict[int, str]:
    """Earliest feed-event date per document.

    A verdict carries no date of its own — the model is asked *whether*, not
    *when*. The feed row that surfaced the article does carry one, so the date
    comes from there rather than from when we happened to scrape it.
    """
    dates: dict[int, str] = {}
    if not document_ids:
        return dates
    for event in session.scalars(
        select(FeedEvent).where(
            FeedEvent.document_id.in_(document_ids), FeedEvent.occurred_on.is_not(None)
        )
    ):
        if event.document_id is None or not event.occurred_on:
            continue
        current = dates.get(event.document_id)
        if current is None or event.occurred_on < current:
            dates[event.document_id] = event.occurred_on
    return dates


def _normalize_day(raw: str | None) -> str | None:
    """GDELT dates arrive as YYYYMMDD; emit ISO so consumers can parse them."""
    if not raw:
        return None
    digits = raw.strip()
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return digits


def detections(
    session: Session,
    run_id: int,
    *,
    min_confidence: float | None = None,
    verified_only: bool = False,
) -> list[Detection]:
    """Every matched pair for a run, with its verification status."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")

    concepts = {
        c.id: c.name
        for c in session.scalars(select(Concept).where(Concept.ontology_id == run.ontology_id))
    }
    loci = {locus.id: locus for locus in session.scalars(select(Locus))}
    labels = {
        (label.document_id, label.concept_id): label
        for label in session.scalars(select(PairLabel))
    }

    matched = list(
        session.scalars(
            select(Verdict).where(
                Verdict.run_id == run_id,
                Verdict.matched.is_(True),
                Verdict.error.is_(None),
            )
        )
    )
    dates = _document_dates(session, [v.document_id for v in matched])

    rows: list[Detection] = []
    for verdict in matched:
        if min_confidence is not None and (verdict.confidence or 0.0) < min_confidence:
            continue

        label = labels.get((verdict.document_id, verdict.concept_id))
        if label is None or label.source not in TRUSTED_SOURCES:
            verification = UNVERIFIED
        else:
            verification = CONFIRMED if label.matched else REJECTED
        if verified_only and verification != CONFIRMED:
            continue

        document = session.get(Document, verdict.document_id)
        locus = loci.get(verdict.locus_id) if verdict.locus_id else None
        rows.append(
            Detection(
                run_id=run_id,
                concept=concepts.get(verdict.concept_id, "?"),
                locus_iso3=locus.iso3 if locus else None,
                locus_iso2=locus.iso2 if locus else None,
                occurred_on=_normalize_day(dates.get(verdict.document_id)),
                document_id=verdict.document_id,
                document_url=document.url if document else "",
                document_title=document.title if document else None,
                confidence=verdict.confidence,
                vote_fraction=verdict.vote_fraction,
                evidence=verdict.evidence,
                verification=verification,
                model=verdict.model,
                prompt_id=verdict.prompt_id,
                ontology_version=run.ontology_version,
            )
        )

    rows.sort(key=lambda d: (d.concept, d.occurred_on or "", -(d.confidence or 0.0)))
    return rows


def events(session: Session, run_id: int, **kwargs) -> list[Event]:
    """Detections collapsed to one row per (concept, locus, date)."""
    grouped: dict[tuple[str, str | None, str | None], Event] = {}
    for detection in detections(session, run_id, **kwargs):
        key = (detection.concept, detection.locus_iso3, detection.occurred_on)
        event = grouped.setdefault(
            key,
            Event(
                run_id=run_id,
                concept=detection.concept,
                locus_iso3=detection.locus_iso3,
                occurred_on=detection.occurred_on,
            ),
        )
        event.document_ids.add(detection.document_id)
        if detection.verification == CONFIRMED:
            event.confirmed += 1
        elif detection.verification == REJECTED:
            event.rejected += 1
        if detection.confidence is not None and (
            event.max_confidence is None or detection.confidence > event.max_confidence
        ):
            event.max_confidence = detection.confidence
        if not event.sample_url:
            event.sample_url = detection.document_url

    rows = list(grouped.values())
    rows.sort(key=lambda e: (e.concept, e.occurred_on or "", -len(e.document_ids)))
    return rows


def _to_csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def export_detections(session: Session, run_id: int, **kwargs) -> str:
    return _to_csv(
        [d.as_row() for d in detections(session, run_id, **kwargs)], DETECTION_COLUMNS
    )


def export_events(session: Session, run_id: int, **kwargs) -> str:
    return _to_csv([e.as_row() for e in events(session, run_id, **kwargs)], EVENT_COLUMNS)


def summary(session: Session, run_id: int) -> dict:
    """Counts by verification status — how much of the output stands up."""
    rows = detections(session, run_id)
    grouped = events(session, run_id)
    by_status: dict[str, int] = defaultdict(int)
    for detection in rows:
        by_status[detection.verification] += 1

    return {
        "run_id": run_id,
        "detections": len(rows),
        "events": len(grouped),
        "by_verification": dict(by_status),
        "concepts": len({d.concept for d in rows}),
        "loci": len({d.locus_iso3 for d in rows if d.locus_iso3}),
        "dated": sum(1 for d in rows if d.occurred_on),
    }
