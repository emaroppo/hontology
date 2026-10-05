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

import logging
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Document, FeedSlice, Run, Verdict
from hontology.evalkit.calendar import Entry, window_documents
from hontology.ingest import dedup, scrape, service
from hontology.judge import run as judge_module
from hontology.retrieve import candidates as candidates_module

log = logging.getLogger(__name__)

SLICES_PER_DAY = 96
SCRAPE_BATCH = 200


def window_status(
    session: Session, entry: Entry, places: list[int], *, before: int, after: int
) -> tuple[bool, list[tuple[str, str]]]:
    """Whether the window is fully ingested, and which slices failed.

    A GKG slice counts only if its scope covers the entry's places: a slice kept
    for another country is no evidence about this one.
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
            elif status == "failed":
                failed.append((feed, key))
        if done < needed:
            complete = False
    return complete, failed


def process_entry(
    session: Session,
    run: Run,
    entry: Entry,
    places: list[int],
    passed: set[int] | None,
    *,
    before: int,
    after: int,
    budget: int | None,
    judge: bool = True,
) -> dict:
    """Scrape, deduplicate, retrieve and judge one ready window. Idempotent.

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
    allowed = window & passed if passed is not None else window

    fetched_now = 0
    while True:
        result = scrape.scrape_pending(
            session, limit=SCRAPE_BATCH, document_ids=sorted(allowed)
        )
        session.commit()
        if not result["attempted"]:
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
        "representatives": len(representatives),
        "copies_marked": dedup_result["newly_marked"],
        "retrieved_new": len(new),
        "retrieval": retrieval,
        "pairs_selected": pairs,
        "judge": judged_now,
    }
    log.info("calendar entry %s: %s", entry.id, summary)
    return summary
