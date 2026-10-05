"""Event calendars: known events, their precursors, and quiet control days.

A calendar is a reviewed CSV of things that are known, from outside the
pipeline, to have happened or not happened: "a port strike in the Netherlands on
2025-10-08", "no port closure in Hong Kong on 2025-12-17". It needs no labelled
articles, which is what makes it the fastest route to a first number: event
recall on the positives, a false-alarm rate on the controls, and lead time where
a precursor was detected before its disruption.

Every entry is scored through the same stages a document passes, so a miss says
where it happened:

    in_feed → passed_filter → fetched → retrieved → matched

An event whose articles never reached the feed needs a different fix from one
the judge rejected, and a calendar that only reported detected/missed would
hide which.

A **window** is the entry's country (or any of its listed countries) over a span
of days around its date, wide enough for coverage that lags the event. A
document belongs to a window when a feed record ties it to one of those places
within that span.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    Candidate,
    Concept,
    Document,
    FeedArticle,
    FeedEvent,
    FeedSlice,
    Locus,
    Run,
    Verdict,
)
from hontology.evalkit.metrics import wilson
from hontology.ingest import filter as ingest_filter

POSITIVE = "positive"
PRECURSOR = "precursor"
CONTROL = "control"
KINDS = (POSITIVE, PRECURSOR, CONTROL)

# Coverage lags events, and a precursor is often reported the day before it is
# formalised, so a window opens a day early and stays open two days late.
DEFAULT_BEFORE = 1
DEFAULT_AFTER = 2

STAGES = ("in_feed", "passed_filter", "fetched", "retrieved", "matched")


class CalendarError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str
    concept: str
    # Several when an event's country is genuinely ambiguous (a pipeline that
    # crosses borders, an attack at sea), written "HUN|SVK" in the file.
    countries: tuple[str, ...]
    date: date
    precursor_of: str | None
    description: str

    def window(self, before: int, after: int) -> tuple[datetime, datetime]:
        """``[start, end)`` in UTC, whole days."""
        start = datetime.combine(self.date - timedelta(days=before), time(), tzinfo=UTC)
        end = datetime.combine(self.date + timedelta(days=after + 1), time(), tzinfo=UTC)
        return start, end


def load(path: Path) -> list[Entry]:
    """Read and validate a calendar file."""
    entries: list[Entry] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            kind = (row.get("kind") or "").strip()
            if kind not in KINDS:
                raise CalendarError(f"line {line}: kind {kind!r} is not one of {KINDS}")
            countries = tuple(
                c.strip().upper()
                for c in (row.get("country_iso3") or "").split("|")
                if c.strip()
            )
            if not countries:
                raise CalendarError(f"line {line}: no country")
            try:
                day = date.fromisoformat((row.get("date") or "").strip())
            except ValueError as exc:
                raise CalendarError(f"line {line}: bad date {row.get('date')!r}") from exc
            entries.append(
                Entry(
                    id=row["id"].strip(),
                    kind=kind,
                    concept=row["concept"].strip(),
                    countries=countries,
                    date=day,
                    precursor_of=(row.get("precursor_of") or "").strip() or None,
                    description=(row.get("description") or "").strip(),
                )
            )

    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise CalendarError("duplicate ids in calendar")
    dangling = [e.id for e in entries if e.precursor_of and e.precursor_of not in set(ids)]
    if dangling:
        raise CalendarError(f"precursor_of points nowhere for: {', '.join(dangling)}")
    return entries


def loci_for(session: Session, entries: list[Entry]) -> dict[str, int]:
    """``{ISO3: locus id}`` for every country the calendar names."""
    wanted = {c for e in entries for c in e.countries}
    found = {
        locus.iso3: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso3.in_(wanted)))
    }
    missing = sorted(wanted - set(found))
    if missing:
        raise CalendarError(f"unknown country codes: {', '.join(missing)}")
    return found


def days_to_ingest(
    entries: list[Entry], loci: dict[str, int], *, before: int, after: int
) -> dict[date, list[int]]:
    """Each day any window touches, with the places that day must keep."""
    days: dict[date, set[int]] = defaultdict(set)
    for entry in entries:
        for offset in range(-before, after + 1):
            days[entry.date + timedelta(days=offset)].update(loci[c] for c in entry.countries)
    return {day: sorted(ids) for day, ids in sorted(days.items())}


def window_documents(
    session: Session, entry: Entry, loci: dict[str, int], *, before: int, after: int
) -> dict[int, datetime]:
    """Documents tied to the entry's places within its window, with first sighting.

    Both feeds count: an event row geolocated there, or a knowledge-graph record
    mentioning it. The sighting time is what lead time is measured from.
    """
    start, end = entry.window(before, after)
    places = [loci[c] for c in entry.countries]
    seen: dict[int, datetime] = {}

    def note(document_id: int | None, moment: datetime | None) -> None:
        if document_id is None or moment is None:
            return
        if document_id not in seen or moment < seen[document_id]:
            seen[document_id] = moment

    for document_id, moment in session.execute(
        select(FeedEvent.document_id, FeedSlice.sliced_at)
        .join(FeedSlice, FeedSlice.id == FeedEvent.slice_id)
        .where(
            FeedEvent.locus_id.in_(places),
            FeedSlice.sliced_at >= start,
            FeedSlice.sliced_at < end,
        )
    ):
        note(document_id, moment)

    for document_id, moment in session.execute(
        select(FeedArticle.document_id, FeedArticle.published_at).where(
            FeedArticle.locus_ids.overlap(places),
            FeedArticle.published_at >= start,
            FeedArticle.published_at < end,
        )
    ):
        note(document_id, moment)
    return seen


def all_window_documents(
    session: Session, entries: list[Entry], *, before: int, after: int
) -> set[int]:
    loci = loci_for(session, entries)
    out: set[int] = set()
    for entry in entries:
        out.update(window_documents(session, entry, loci, before=before, after=after))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class EntryResult:
    entry: Entry
    stages: dict[str, int] = field(default_factory=dict)
    first_match: datetime | None = None

    @property
    def detected(self) -> bool:
        return self.stages.get("matched", 0) > 0

    @property
    def lost_at(self) -> str | None:
        """The first stage that reached zero: where this entry was lost."""
        for stage in STAGES:
            if self.stages.get(stage, 0) == 0:
                return stage
        return None

    def as_dict(self) -> dict:
        return {
            "id": self.entry.id,
            "kind": self.entry.kind,
            "concept": self.entry.concept,
            "countries": "|".join(self.entry.countries),
            "date": self.entry.date.isoformat(),
            "detected": self.detected,
            "lost_at": self.lost_at,
            "first_match": self.first_match.isoformat() if self.first_match else None,
            **self.stages,
        }


def _rate(hits: int, n: int) -> dict:
    low, high = wilson(hits, n)
    return {"hits": hits, "n": n, "rate": (hits / n) if n else None, "ci": [low, high]}


def evaluate(
    session: Session,
    run_id: int,
    entries: list[Entry],
    *,
    before: int = DEFAULT_BEFORE,
    after: int = DEFAULT_AFTER,
) -> dict:
    """Score one run against a calendar, entry by entry and in aggregate."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")

    concepts = {
        c.name: c.id
        for c in session.scalars(select(Concept).where(Concept.ontology_id == run.ontology_id))
    }
    unknown = sorted({e.concept for e in entries} - set(concepts))
    if unknown:
        raise CalendarError(f"concepts not in run's ontology: {', '.join(unknown)}")

    loci = loci_for(session, entries)
    passed = set(ingest_filter.matching_documents(session, run.ontology_id))
    has_filter = bool(passed)

    results: list[EntryResult] = []
    for entry in entries:
        concept_id = concepts[entry.concept]
        docs = window_documents(session, entry, loci, before=before, after=after)
        filtered = {d for d in docs if d in passed} if has_filter else set(docs)
        fetched = (
            set(
                session.scalars(
                    select(Document.id).where(
                        Document.id.in_(filtered),
                        Document.body_path.is_not(None),
                        Document.is_junk.is_(False),
                    )
                )
            )
            if filtered
            else set()
        )
        retrieved = (
            set(
                session.scalars(
                    select(Candidate.document_id).where(
                        Candidate.run_id == run_id,
                        Candidate.concept_id == concept_id,
                        Candidate.selected.is_(True),
                        Candidate.document_id.in_(fetched),
                    )
                )
            )
            if fetched
            else set()
        )
        matched = (
            set(
                session.scalars(
                    select(Verdict.document_id).where(
                        Verdict.run_id == run_id,
                        Verdict.concept_id == concept_id,
                        Verdict.matched.is_(True),
                        Verdict.document_id.in_(retrieved),
                    )
                )
            )
            if retrieved
            else set()
        )
        results.append(
            EntryResult(
                entry=entry,
                stages={
                    "in_feed": len(docs),
                    "passed_filter": len(filtered),
                    "fetched": len(fetched),
                    "retrieved": len(retrieved),
                    "matched": len(matched),
                },
                first_match=min((docs[d] for d in matched), default=None),
            )
        )

    by_id = {r.entry.id: r for r in results}
    lead_times = []
    for result in results:
        if result.entry.kind != PRECURSOR or not result.detected:
            continue
        target = by_id[result.entry.precursor_of]
        start = datetime.combine(target.entry.date, time(), tzinfo=UTC)
        lead_times.append(
            {
                "precursor": result.entry.id,
                "disruption": target.entry.id,
                # Positive = the precursor was seen before the disruption's day.
                "lead_days": round((start - result.first_match).total_seconds() / 86400, 2),
                "disruption_detected": target.detected,
            }
        )

    def of(kind: str) -> list[EntryResult]:
        return [r for r in results if r.entry.kind == kind]

    def lost(kind: str) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGES}
        for r in of(kind):
            if r.lost_at:
                counts[r.lost_at] += 1
        return counts

    return {
        "run_id": run_id,
        "ontology_id": run.ontology_id,
        "window": {"before_days": before, "after_days": after},
        "filter_applied": has_filter,
        "summary": {
            "event_recall": _rate(sum(r.detected for r in of(POSITIVE)), len(of(POSITIVE))),
            "precursor_recall": _rate(
                sum(r.detected for r in of(PRECURSOR)), len(of(PRECURSOR))
            ),
            "false_alarm_rate": _rate(sum(r.detected for r in of(CONTROL)), len(of(CONTROL))),
            "positives_lost_at": lost(POSITIVE),
        },
        "lead_times": lead_times,
        "entries": [r.as_dict() for r in results],
    }
