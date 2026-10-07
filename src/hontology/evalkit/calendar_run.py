"""Running a calendar a window at a time, as its feed arrives.

A calendar backfill takes hours, and the run that scores it should not have to
wait for the last day. An entry is *ready* once every slice of its window is
ingested in both feeds; from then on its documents are fixed, so it can be
scraped, deduplicated, retrieved and judged without waiting for any other.

All entries feed **one run**, so the result is scored once with
`evalkit.calendar.evaluate`. Retrieval appends rather than rebuilds, and the
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
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Concept, Document, FeedSlice, Run, Verdict
from hontology.evalkit.calendar import Entry, window_documents
from hontology.ingest import dedup, scrape, service
from hontology.ingest import filter as ingest_filter
from hontology.ingest.dedup import representative_of
from hontology.judge import run as judge_module
from hontology.ontology import hierarchy, snapshots
from hontology.retrieve import candidates as candidates_module

log = logging.getLogger(__name__)

SLICES_PER_DAY = 96
SCRAPE_BATCH = 200


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
    for feed in (service.FEED, service.FEED_GKG):
        rows = session.execute(
            select(FeedSlice.slice_key, FeedSlice.status, FeedSlice.scope).where(
                FeedSlice.feed == feed,
                FeedSlice.sliced_at >= start,
                FeedSlice.sliced_at < end,
            )
        ).all()
        done = 0
        for key, status, scope in rows:
            covers = feed != service.FEED_GKG or scope is None or set(places) <= set(scope)
            if status in service.TERMINAL and covers:
                done += 1
            elif status == "failed" or status in service.TERMINAL:
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
    start, end = entry.window(before, after)
    for feed in (service.FEED, service.FEED_GKG):
        for status, scope in session.execute(
            select(FeedSlice.status, FeedSlice.scope).where(
                FeedSlice.feed == feed,
                FeedSlice.sliced_at >= start,
                FeedSlice.sliced_at < end,
            )
        ):
            covers = feed != service.FEED_GKG or scope is None or set(places) <= set(scope)
            if covers and status in service.TERMINAL and status != "missing":
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
        # Another run's retrieval, reused: judge exactly the documents it
        # judged in this window, with no scraping, deduplication or retrieval
        # of our own, so the two runs differ only in how they judge.
        representatives = copy_window_candidates(session, candidates_from, run.id, window)
        session.commit()
        allowed = window
        fetched_now = deferred = 0
        dedup_result = {"newly_marked": 0}
        new: list[Document] = []
        retrieval = {"copied_from_run": candidates_from, "documents": len(representatives)}
    else:
        # Filter matches are computed for this window now, not taken from a set
        # computed earlier: a window that finished ingesting after that set was
        # built would otherwise be judged against articles that did not exist yet,
        # find nothing in scope, and be marked done.
        ontology_id = run.ontology_id
        has_links = bool(
            ingest_filter.concept_code_map(session, ontology_id)
            or ingest_filter.concept_theme_map(session, ontology_id)
        )
        if has_links:
            matches = ingest_filter.matching_documents(
                session, ontology_id, document_ids=sorted(window)
            )
            allowed = window & set(matches)
        else:
            allowed = window  # no links at all: the filter cannot distinguish anything

        fetched_now = 0
        deferred = 0
        while True:
            result = scrape.scrape_pending(
                session, limit=SCRAPE_BATCH, document_ids=sorted(allowed)
            )
            session.commit()
            deferred = result.get("deferred", 0)
            if not result["attempted"]:
                # Nothing left we could claim. A scraper running ahead may
                # still hold some of this window's documents: wait for it,
                # then look again, so none is retrieved before it is fetched.
                if scrape.wait_for_claimed(session, sorted(allowed)):
                    continue
                break
            fetched_now += result["attempted"]

        dedup_result = dedup.deduplicate(session, sorted(allowed))
        session.commit()

        representatives = set(
            session.scalars(
                select(Document.id).where(
                    Document.id.in_(allowed),
                    Document.body_path.is_not(None),
                    Document.is_junk.is_(False),
                    Document.duplicate_of.is_(None),
                )
            )
        )
        already = set(
            session.scalars(
                select(Candidate.document_id)
                .where(Candidate.run_id == run.id, Candidate.document_id.in_(representatives))
                .distinct()
            )
        )
        new = list(
            session.scalars(select(Document).where(Document.id.in_(representatives - already)))
        )
        retrieval = None
        if new:
            retrieval = candidates_module.build_semantic(
                session,
                run.id,
                ontology_id=run.ontology_id,
                documents=new,
                config=run.config["candidates"],
                embed_body_limit=run.config["common"]["embed_body_limit"],
            ).as_dict()
            session.commit()

    remaining = None
    if budget is not None:
        judged = (
            session.scalar(
                select(func.count(Verdict.id)).where(
                    Verdict.run_id == run.id, Verdict.document_id.in_(representatives)
                )
            )
            or 0
        )
        remaining = max(0, budget - judged)
    pairs = (
        session.scalar(
            select(func.count(Candidate.id)).where(
                Candidate.run_id == run.id,
                Candidate.selected.is_(True),
                Candidate.document_id.in_(representatives),
            )
        )
        or 0
    )
    judged_now = None
    if judge and (remaining is None or remaining > 0):
        judged_now = judge_module.judge_run(
            session,
            run.id,
            config=run.config,
            judge_body_limit=run.config["common"]["judge_body_limit"],
            limit=remaining,
            document_ids=representatives,
            by_score=budget is not None,
        )
        session.commit()

    summary = {
        "entry": entry.id,
        "in_window": len(window),
        "passed_filter": len(allowed),
        "fetched_now": fetched_now,
        # Still pending at the end: hosts whose crawl delay outlasted the pass.
        "deferred": deferred,
        "representatives": len(representatives),
        "copies_marked": dedup_result["newly_marked"],
        "retrieved_new": len(new),
        "retrieval": retrieval,
        "pairs_selected": pairs,
        "judge": judged_now,
    }
    log.info("calendar entry %s: %s", entry.id, summary)
    return summary


def copy_window_candidates(
    session: Session, source_run_id: int, target_run_id: int, window: set[int]
) -> set[int]:
    """Copy *source*'s candidates for the window's representatives into *target*.

    Returns the representatives *source* retrieved for. Documents already
    copied are left alone, so this is safe to repeat.
    """
    representatives = (
        set(representative_of(session, sorted(window)).values()) if window else set()
    )
    if not representatives:
        return set()
    retrieved = set(
        session.scalars(
            select(Candidate.document_id)
            .where(
                Candidate.run_id == source_run_id,
                Candidate.document_id.in_(representatives),
            )
            .distinct()
        )
    )
    present = set(
        session.scalars(
            select(Candidate.document_id)
            .where(Candidate.run_id == target_run_id, Candidate.document_id.in_(retrieved))
            .distinct()
        )
    )
    for row in session.scalars(
        select(Candidate).where(
            Candidate.run_id == source_run_id, Candidate.document_id.in_(retrieved - present)
        )
    ):
        session.add(
            Candidate(
                run_id=target_run_id,
                document_id=row.document_id,
                concept_id=row.concept_id,
                source=row.source,
                score=row.score,
                rank=row.rank,
                selected=row.selected,
                matched_code=row.matched_code,
                matched_level=row.matched_level,
            )
        )
    session.flush()
    return retrieved


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
