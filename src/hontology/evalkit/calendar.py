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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import (
    CalendarReview,
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
from hontology.ingest.dedup import representative_of
from hontology.judge import prompts

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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class EntryResult:
    entry: Entry
    stages: dict[str, int] = field(default_factory=dict)
    first_match: datetime | None = None
    # Fetched documents left after near-duplicates collapse onto one
    # representative: what retrieval and the judge actually see.
    unique: int = 0
    # Judging spent on this window's documents, across every concept.
    cost: dict = field(default_factory=dict)
    # One row per matched representative, earliest sighting first, with any
    # review of it: {url, first_seen, evidence, confirmed}.
    matches: list[dict] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.stages.get("matched", 0) > 0

    @property
    def verified(self) -> bool | None:
        """Whether a person confirmed a match: True, False, or None if pending.

        For an event or precursor, True means some matched article describes
        this very event, and False that every match was about something else.
        For a control, True means a match reported a real instance, so the
        control itself was wrong; False confirms the false alarm.
        """
        if not self.matches:
            return False
        reviews = [m["confirmed"] for m in self.matches]
        if any(r is True for r in reviews):
            return True
        if all(r is False for r in reviews):
            return False
        return None

    @property
    def first_confirmed(self) -> datetime | None:
        return min(
            (m["first_seen"] for m in self.matches if m["confirmed"] is True), default=None
        )

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
            "verified": self.verified,
            "matches": [
                m | {"first_seen": m["first_seen"].isoformat() if m["first_seen"] else None}
                for m in self.matches
            ],
            "unique": self.unique,
            "cost": self.cost,
            **self.stages,
        }


def _rate(hits: int, n: int) -> dict:
    low, high = wilson(hits, n)
    return {"hits": hits, "n": n, "rate": (hits / n) if n else None, "ci": [low, high]}


def judging_cost(session: Session, run_id: int, document_ids: set[int] | None = None) -> dict:
    """Pairs judged, tokens and seconds a run spent, optionally on some documents.

    Read from the verdicts themselves, so it is the same measure in every arm:
    a hierarchical run's internal-class answers are verdict rows too.
    """
    query = select(
        func.count(Verdict.id),
        func.coalesce(func.sum(Verdict.input_tokens), 0),
        func.coalesce(func.sum(Verdict.output_tokens), 0),
        func.coalesce(func.sum(Verdict.latency_s), 0.0),
    ).where(Verdict.run_id == run_id)
    if document_ids is not None:
        if not document_ids:
            return {"pairs": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        query = query.where(Verdict.document_id.in_(document_ids))
    pairs, tokens_in, tokens_out, seconds = session.execute(query).one()
    return {
        "pairs": int(pairs),
        "input_tokens": int(tokens_in),
        "output_tokens": int(tokens_out),
        "seconds": round(float(seconds), 1),
    }


def _matches(
    session: Session,
    run_id: int,
    concept_id: int,
    entry: Entry,
    matched: set[int],
    reads: dict[int, int],
    docs: dict[int, datetime],
    reviews: dict[tuple[str, str], bool],
) -> list[dict]:
    """The entry's matched representatives, earliest first, with their reviews.

    One row per representative, not per copy: reviewing a story once covers
    every outlet that republished it.
    """
    seen: dict[int, datetime] = {}
    for doc_id in matched:
        rep = reads[doc_id]
        if rep not in seen or docs[doc_id] < seen[rep]:
            seen[rep] = docs[doc_id]
    if not seen:
        return []
    urls: dict[int, str] = {
        doc_id: url
        for doc_id, url in session.execute(
            select(Document.id, Document.url).where(Document.id.in_(seen))
        )
    }
    evidence: dict[int, str | None] = {
        doc_id: text
        for doc_id, text in session.execute(
            select(Verdict.document_id, Verdict.evidence).where(
                Verdict.run_id == run_id,
                Verdict.concept_id == concept_id,
                Verdict.document_id.in_(seen),
            )
        )
    }
    rows = [
        {
            "url": urls[rep],
            "first_seen": seen[rep],
            "evidence": evidence.get(rep),
            "confirmed": reviews.get((entry.id, urls[rep])),
        }
        for rep in seen
    ]
    return sorted(rows, key=lambda row: row["first_seen"])


def processed_ids(run: Run, entries: list[Entry]) -> set[str]:
    """The entries this run has actually been through.

    A run made with `run calendar` records each entry as it finishes; one made
    with `run start --calendar` processed every window at once, so its entries
    count only once it is done. Anything else is not scored: an entry the run
    never reached has no verdicts, which is indistinguishable from a miss.
    """
    ids = {e.id for e in entries}
    done = (run.manifest or {}).get("calendar_done")
    if done is not None:
        return ids & set(done)
    return ids if run.status == "done" else set()


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
    processed = processed_ids(run, entries)
    # The filter is matched against the calendar's own windows: over the whole
    # corpus it reads every feed record, which is slow and was what exhausted
    # memory when the corpus held millions of events.
    has_filter = bool(
        ingest_filter.concept_code_map(session, run.ontology_id)
        or ingest_filter.concept_theme_map(session, run.ontology_id)
    )
    in_windows: set[int] = set()
    for entry in entries:
        if entry.id in processed:
            in_windows |= set(
                window_documents(session, entry, loci, before=before, after=after)
            )
    passed = (
        set(
            ingest_filter.matching_documents(
                session, run.ontology_id, document_ids=sorted(in_windows)
            )
        )
        if has_filter and in_windows
        else set()
    )
    hierarchical = (
        prompts.get(run.config["judge"]["prompt_id"]).mode == prompts.HIERARCHICAL
        if run.config.get("judge")
        else False
    )
    reviews = {
        (review.entry_id, review.document_url): review.confirmed
        for review in session.scalars(
            select(CalendarReview).where(CalendarReview.entry_id.in_([e.id for e in entries]))
        )
    }

    results: list[EntryResult] = []
    for entry in entries:
        if entry.id not in processed:
            continue
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
        # A near-duplicate is never retrieved or judged itself; it reads its
        # representative's results, wherever that representative was seen.
        reads = representative_of(session, sorted(fetched)) if fetched else {}
        # A flat run judges a concept only where retrieval selected it. A
        # hierarchical run judges every document retrieval let through, and
        # reaches the concept by descending, so there "retrieved" means any
        # selected leaf at all.
        retrieval_gate = [
            Candidate.run_id == run_id,
            Candidate.selected.is_(True),
            Candidate.document_id.in_(set(reads.values())),
        ]
        if not hierarchical:
            retrieval_gate.append(Candidate.concept_id == concept_id)
        retrieved_reps = (
            set(session.scalars(select(Candidate.document_id).where(*retrieval_gate)))
            if reads
            else set()
        )
        verdicts = (
            session.execute(
                select(Verdict.document_id, Verdict.matched).where(
                    Verdict.run_id == run_id,
                    Verdict.concept_id == concept_id,
                    Verdict.document_id.in_(retrieved_reps),
                )
            ).all()
            if retrieved_reps
            else []
        )
        judged_reps = {doc_id for doc_id, _ in verdicts}
        matched_reps = {doc_id for doc_id, matched in verdicts if matched}
        retrieved = {d for d, rep in reads.items() if rep in retrieved_reps}
        judged = {d for d, rep in reads.items() if rep in judged_reps}
        matched = {d for d, rep in reads.items() if rep in matched_reps}
        results.append(
            EntryResult(
                entry=entry,
                stages={
                    "in_feed": len(docs),
                    "passed_filter": len(filtered),
                    "fetched": len(fetched),
                    "retrieved": len(retrieved),
                    "judged": len(judged),
                    "matched": len(matched),
                },
                first_match=min((docs[d] for d in matched), default=None),
                unique=len(set(reads.values())),
                cost=judging_cost(session, run_id, set(reads.values())),
                matches=_matches(
                    session, run_id, concept_id, entry, matched, reads, docs, reviews
                ),
            )
        )

    by_id = {r.entry.id: r for r in results}
    lead_times = []
    for result in results:
        if (
            result.entry.kind != PRECURSOR
            or result.entry.precursor_of is None
            or result.first_match is None
        ):
            continue
        target = by_id.get(result.entry.precursor_of)
        if target is None:
            # The disruption itself was not processed: no lead time to speak of.
            continue
        start = datetime.combine(target.entry.date, time(), tzinfo=UTC)
        lead_times.append(
            {
                "precursor": result.entry.id,
                "disruption": target.entry.id,
                # Positive = the precursor was seen before the disruption's day.
                "lead_days": round((start - result.first_match).total_seconds() / 86400, 2),
                # The same, from the earliest match a person confirmed.
                "verified_lead_days": (
                    round((start - result.first_confirmed).total_seconds() / 86400, 2)
                    if result.first_confirmed
                    else None
                ),
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
        "not_processed": [e.id for e in entries if e.id not in processed],
        "summary": {
            "event_recall": _rate(sum(r.detected for r in of(POSITIVE)), len(of(POSITIVE))),
            "precursor_recall": _rate(
                sum(r.detected for r in of(PRECURSOR)), len(of(PRECURSOR))
            ),
            "false_alarm_rate": _rate(sum(r.detected for r in of(CONTROL)), len(of(CONTROL))),
            "positives_lost_at": lost(POSITIVE),
            "verified": {
                "event_recall": _rate(
                    sum(r.verified is True for r in of(POSITIVE)), len(of(POSITIVE))
                ),
                "precursor_recall": _rate(
                    sum(r.verified is True for r in of(PRECURSOR)), len(of(PRECURSOR))
                ),
                # A control whose match reported a real instance is a calendar
                # error: withdrawn from the denominator, and listed.
                "false_alarm_rate": _rate(
                    sum(r.detected and r.verified is False for r in of(CONTROL)),
                    sum(r.verified is not True for r in of(CONTROL)),
                ),
                "controls_withdrawn": [r.entry.id for r in of(CONTROL) if r.verified is True],
                "pending_review": [r.entry.id for r in results if r.verified is None],
            },
        },
        "lead_times": lead_times,
        # The whole run, so windows that share documents are not counted twice.
        "cost": judging_cost(session, run_id),
        "entries": [r.as_dict() for r in results],
    }
