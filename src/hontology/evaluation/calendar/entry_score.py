"""Following one entry's window through the stages, from the feed to a match."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import Candidate, Document, Verdict
from hontology.evaluation.calendar.entry_result import EntryResult
from hontology.evaluation.calendar.events import Entry
from hontology.pipeline.ingest.articles.dedup import representative_of


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


def score_entry(
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
