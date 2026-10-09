"""Event calendars: known events, their precursors, and quiet control days.

A calendar is a reviewed CSV of things that are known, from outside the
pipeline, to have happened or not happened: "a port strike in the Netherlands on
2025-10-08", "no port closure in Hong Kong on 2025-12-17". It needs no labelled
articles, which is what makes it the fastest route to a first number: event
recall on the positives, a false-alarm rate on the controls, and lead time where
a precursor was detected before its disruption.

Every entry is scored through the same stages a document passes, so a miss says
where it happened:

    in_feed → passed_filter → fetched → retrieved → judged → matched

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
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    FeedArticle,
    FeedEvent,
    FeedSlice,
    Locus,
)

POSITIVE = "positive"
PRECURSOR = "precursor"
CONTROL = "control"
KINDS = (POSITIVE, PRECURSOR, CONTROL)

# Coverage lags events, and a precursor is often reported the day before it is
# formalised, so a window opens a day early and stays open two days late.
DEFAULT_BEFORE = 1
DEFAULT_AFTER = 2

# `judged` sits apart from `matched` because a run with a judging budget can
# retrieve a document and never judge it; that miss is the budget's, not the
# judge's, and must not read as a rejection.
STAGES = ("in_feed", "passed_filter", "fetched", "retrieved", "judged", "matched")


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
