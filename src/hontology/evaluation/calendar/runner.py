"""Running a calendar a window at a time, as its feed arrives.

A calendar backfill takes hours, and the run that scores it should not have to
wait for the last day. An entry is *ready* once every slice of its window is
ingested in both feeds; from then on its documents are fixed, so it can be
scraped, deduplicated, retrieved and judged without waiting for any other.

All entries feed **one run**, so the result is scored once with
`evaluation.calendar.score.evaluate`. Retrieval appends rather than rebuilds, and the
judge skips pairs already done, which also makes the whole loop resumable: a
restart re-checks each entry and does only what is missing.

An optional **budget** caps the pairs judged per window, highest retrieval
score first. The judge is never told which concept the calendar expects, so
the cap is blind to the answer; an event lost to it shows up as a retrieved
document that was never judged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, FeedSlice, Run
from hontology.evaluation.calendar.events import Entry, window_documents
from hontology.evaluation.calendar.window_work import (
    copy_window_candidates,
    judge_window,
    prepare_window,
    reuse_retrieval,
)
from hontology.ontology import hierarchy, snapshots
from hontology.pipeline.ingest.feed import slices
from hontology.pipeline.judge import run as judge_module

log = logging.getLogger(__name__)

SLICES_PER_DAY = 96


def _window_slices(
    session: Session, entry: Entry, places: list[int], *, before: int, after: int
) -> Iterator[tuple[str, list[tuple[str, str, bool]]]]:
    """Per feed, the window's slices as ``(key, status, covers)``.

    A GKG slice covers the entry only if its scope includes all of *places*.
    """
    start, end = entry.window(before, after)
    for feed in (slices.FEED, slices.FEED_GKG):
        rows = session.execute(
            select(FeedSlice.slice_key, FeedSlice.status, FeedSlice.scope).where(
                FeedSlice.feed == feed,
                FeedSlice.sliced_at >= start,
                FeedSlice.sliced_at < end,
            )
        )
        gkg = feed == slices.FEED_GKG
        covered = [
            (key, status, not gkg or scope is None or set(places) <= set(scope))
            for key, status, scope in rows
        ]
        yield feed, covered


def window_status(
    session: Session, entry: Entry, places: list[int], *, before: int, after: int
) -> tuple[bool, list[tuple[str, str]]]:
    """Whether the window is fully ingested, and which slices to ingest again.

    A GKG slice counts only if its scope covers the entry's places: a slice kept
    for another country is no evidence about this one. Such a slice is returned
    for ingesting again with this entry's places, as a failed one is; otherwise
    nothing would ever widen it, and the window would wait for good. (Backfills
    running side by side can leave one: each keeps its own countries.)
    """
    start, end = entry.window(before, after)
    needed = int((end - start) / timedelta(days=1)) * SLICES_PER_DAY
    complete = True
    failed: list[tuple[str, str]] = []
    for feed, rows in _window_slices(session, entry, places, before=before, after=after):
        done = 0
        for key, status, covers in rows:
            if status in slices.TERMINAL and covers:
                done += 1
            elif status == "failed" or status in slices.TERMINAL:
                failed.append((feed, key))
        if done < needed:
            complete = False
    return complete, failed


def window_unobservable(
    session: Session, entry: Entry, places: list[int], *, before: int, after: int
) -> bool:
    """Whether the source published nothing for the window, in either feed.

    A slice the source never published is marked missing, which is terminal, so
    a window inside an outage counts as ingested with no articles in it. Scored,
    its event would be a miss the detector never had a chance at, and a control
    would pass for the same reason; such a window is set aside instead.
    """
    for _feed, rows in _window_slices(session, entry, places, before=before, after=after):
        for _key, status, covers in rows:
            if covers and status in slices.TERMINAL and status != "missing":
                return False
    return True


def process_entry(
    session: Session,
    run: Run,
    entry: Entry,
    places: list[int],
    *,
    before: int,
    after: int,
    budget: int | None,
    judge: bool = True,
    candidates_from: int | None = None,
) -> dict:
    """Scrape, deduplicate, retrieve and judge one ready window. Idempotent.

    With *candidates_from*, the window reuses that run's retrieval instead of
    building its own (see `copy_window_candidates`).

    With *judge* off it stops after retrieval, so the volume a window would
    send to the judge can be seen before any judging is paid for.
    """
    window = set(
        window_documents(
            session,
            entry,
            dict(zip(entry.countries, places, strict=True)),
            before=before,
            after=after,
        )
    )
    if candidates_from is not None:
        work = reuse_retrieval(session, run, window, candidates_from)
    else:
        work = prepare_window(session, run, window)
    pairs, judged_now = judge_window(
        session, run, work.representatives, budget=budget, judge=judge
    )

    summary = {
        "entry": entry.id,
        "in_window": len(window),
        "passed_filter": len(work.allowed),
        "fetched_now": work.fetched_now,
        # Still pending at the end: hosts whose crawl delay outlasted the pass.
        "deferred": work.deferred,
        "representatives": len(work.representatives),
        "copies_marked": work.copies_marked,
        "retrieved_new": len(work.new),
        "retrieval": work.retrieval,
        "pairs_selected": pairs,
        "judge": judged_now,
    }
    log.info("calendar entry %s: %s", entry.id, summary)
    return summary


def judge_documents(
    session: Session, run: Run, *, source_run_id: int, document_ids: list[int]
) -> dict:
    """Judge exactly these documents with another run's retrieval. Idempotent.

    For scoring an arm on the labelled sample before, or instead of, judging
    every calendar window: the sample's documents get *source*'s candidates
    and are judged in full, with no budget. Calling it again with more
    documents judges only the new ones.
    """
    retrieved = copy_window_candidates(session, source_run_id, run.id, set(document_ids))
    session.commit()
    result = judge_module.judge_run(
        session,
        run.id,
        config=run.config,
        judge_body_limit=run.config["common"]["judge_body_limit"],
        document_ids=retrieved,
    )
    session.commit()
    return {"documents": len(document_ids), "retrieved": len(retrieved), "judge": result}


def check_same_leaves(session: Session, source_run: Run, ontology_id: int) -> None:
    """Refuse to reuse a run's retrieval if any leaf has been reworded since.

    The source's retrieval and the leaf labels both answered the leaves'
    wording at the source's version; a reworded leaf would make the arms
    answer different questions.
    """
    snapshot = snapshots.get_version(session, ontology_id, source_run.ontology_version)
    if snapshot is None:
        raise ValueError(
            f"run {source_run.id}'s version {source_run.ontology_version} is unknown"
        )
    then = {row["id"]: row for row in json.loads(snapshot.payload)}
    now = {
        row["id"]: row
        for row in snapshots.normalize_set(
            list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
        )
    }
    leaves = hierarchy.leaves(session, ontology_id)
    changed = sorted(cid for cid in leaves if then.get(cid) != now.get(cid))
    if changed:
        raise ValueError(f"leaves differ from run {source_run.id}'s version: {changed}")
