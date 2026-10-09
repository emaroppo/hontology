"""Scoring a run against an event calendar.

Each entry's window is followed through the stages in `calendar.STAGES`, so a
miss says where it happened; the summary gives event recall, precursor recall
and the false-alarm rate on controls, raw and as verified by a person.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import concept_ids_by_name, get_run
from hontology.db.models import CalendarReview, Run
from hontology.evaluation.calendar.entry_score import judging_cost, score_entry
from hontology.evaluation.calendar.events import (
    DEFAULT_AFTER,
    DEFAULT_BEFORE,
    CalendarError,
    Entry,
    loci_for,
    window_documents,
)
from hontology.evaluation.calendar.summary import lead_times, summarize
from hontology.pipeline.ingest.articles import filter as ingest_filter
from hontology.pipeline.judge import prompts


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
        score_entry(
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
        "summary": summarize(results),
        "lead_times": lead_times(results),
        # The whole run, so windows that share documents are not counted twice.
        "cost": judging_cost(session, run_id),
        "entries": [r.as_dict() for r in results],
    }
