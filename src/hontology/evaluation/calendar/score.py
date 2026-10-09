"""Scoring a run against an event calendar.

Each entry's window is followed through the stages in `calendar.STAGES`, so a
miss says where it happened; the summary gives event recall, precursor recall
and the false-alarm rate on controls, raw and as verified by a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.lookups import concept_ids_by_name, get_run
from hontology.db.models import CalendarReview, Candidate, Document, Run, Verdict
from hontology.evaluation.calendar.events import (
    CONTROL,
    DEFAULT_AFTER,
    DEFAULT_BEFORE,
    POSITIVE,
    PRECURSOR,
    STAGES,
    CalendarError,
    Entry,
    loci_for,
    window_documents,
)
from hontology.evaluation.metrics import wilson
from hontology.pipeline.ingest.articles import filter as ingest_filter
from hontology.pipeline.ingest.articles.dedup import representative_of
from hontology.pipeline.judge import prompts


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
        query = query.where(among(Verdict.document_id, document_ids))
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
            select(Document.id, Document.url).where(among(Document.id, seen))
        )
    }
    evidence: dict[int, str | None] = {
        doc_id: text
        for doc_id, text in session.execute(
            select(Verdict.document_id, Verdict.evidence).where(
                Verdict.run_id == run_id,
                Verdict.concept_id == concept_id,
                among(Verdict.document_id, seen),
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


def _score_entry(
    session: Session,
    run_id: int,
    entry: Entry,
    concept_id: int,
    docs: dict[int, datetime],
    *,
    has_filter: bool,
    passed: set[int],
    hierarchical: bool,
    reviews: dict[tuple[str, str], bool],
) -> EntryResult:
    """One entry's window, followed stage by stage from the feed to a match."""
    filtered = {d for d in docs if d in passed} if has_filter else set(docs)
    fetched = (
        set(
            session.scalars(
                select(Document.id).where(
                    among(Document.id, filtered),
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
        among(Candidate.document_id, set(reads.values())),
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
                among(Verdict.document_id, retrieved_reps),
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
    return EntryResult(
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
        matches=_matches(session, run_id, concept_id, entry, matched, reads, docs, reviews),
    )


def _lead_times(results: list[EntryResult]) -> list[dict]:
    """How far ahead of its disruption each detected precursor was seen."""
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
    return lead_times


def _summary(results: list[EntryResult]) -> dict:
    """Recall, false alarms and where positives were lost, raw and verified."""

    def of(kind: str) -> list[EntryResult]:
        return [r for r in results if r.entry.kind == kind]

    def lost(kind: str) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGES}
        for r in of(kind):
            if r.lost_at:
                counts[r.lost_at] += 1
        return counts

    return {
        "event_recall": _rate(sum(r.detected for r in of(POSITIVE)), len(of(POSITIVE))),
        "precursor_recall": _rate(sum(r.detected for r in of(PRECURSOR)), len(of(PRECURSOR))),
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
    }


def evaluate(
    session: Session,
    run_id: int,
    entries: list[Entry],
    *,
    before: int = DEFAULT_BEFORE,
    after: int = DEFAULT_AFTER,
) -> dict:
    """Score one run against a calendar, entry by entry and in aggregate."""
    run = get_run(session, run_id)

    concepts = concept_ids_by_name(session, run.ontology_id)
    unknown = sorted({e.concept for e in entries} - set(concepts))
    if unknown:
        raise CalendarError(f"concepts not in run's ontology: {', '.join(unknown)}")

    loci = loci_for(session, entries)
    processed = processed_ids(run, entries)
    # The filter is matched against the calendar's own windows: over the whole
    # corpus it reads every feed record, which is slow and was what exhausted
    # memory when the corpus held millions of events.
    has_filter = ingest_filter.has_links(session, run.ontology_id)
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
        prompts.get(run.config["judge"]["prompt_id"]).mode in prompts.TOP_DOWN_MODES
        if run.config.get("judge")
        else False
    )
    reviews = {
        (review.entry_id, review.document_url): review.confirmed
        for review in session.scalars(
            select(CalendarReview).where(CalendarReview.entry_id.in_([e.id for e in entries]))
        )
    }

    results = [
        _score_entry(
            session,
            run_id,
            entry,
            concepts[entry.concept],
            window_documents(session, entry, loci, before=before, after=after),
            has_filter=has_filter,
            passed=passed,
            hierarchical=hierarchical,
            reviews=reviews,
        )
        for entry in entries
        if entry.id in processed
    ]
    return {
        "run_id": run_id,
        "ontology_id": run.ontology_id,
        "window": {"before_days": before, "after_days": after},
        "filter_applied": has_filter,
        "not_processed": [e.id for e in entries if e.id not in processed],
        "summary": _summary(results),
        "lead_times": _lead_times(results),
        # The whole run, so windows that share documents are not counted twice.
        "cost": judging_cost(session, run_id),
        "entries": [r.as_dict() for r in results],
    }
