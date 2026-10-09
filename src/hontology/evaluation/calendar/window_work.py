"""The steps one ready calendar window goes through before it is judged.

Either the window reuses another run's retrieval, or it is filtered, scraped,
deduplicated and retrieved here; then the selected pairs are judged, within
the budget if there is one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.lookups import count
from hontology.db.models import Candidate, Document, Run, Verdict
from hontology.pipeline.ingest.articles import dedup, scrape
from hontology.pipeline.ingest.articles import filter as ingest_filter
from hontology.pipeline.ingest.articles.dedup import representative_of
from hontology.pipeline.judge import run as judge_module
from hontology.pipeline.retrieve import candidates as candidates_module
from hontology.pipeline.runs import link_versions, versions
from hontology.pipeline.runs.reuse import clone_candidates

SCRAPE_BATCH = 200


@dataclass
class WindowWork:
    """What preparing a window did, up to and including retrieval."""

    allowed: set[int]
    representatives: set[int]
    retrieval: dict | None
    fetched_now: int = 0
    deferred: int = 0
    copies_marked: int = 0
    new: list[Document] = field(default_factory=list)


def reuse_retrieval(
    session: Session, run: Run, window: set[int], candidates_from: int
) -> WindowWork:
    """Another run's retrieval, reused: judge exactly the documents it judged
    in this window, with no scraping, deduplication or retrieval of our own,
    so the two runs differ only in how they judge."""
    representatives = copy_window_candidates(session, candidates_from, run.id, window)
    session.commit()
    return WindowWork(
        allowed=window,
        representatives=representatives,
        retrieval={"copied_from_run": candidates_from, "documents": len(representatives)},
    )


def prepare_window(session: Session, run: Run, window: set[int]) -> WindowWork:
    """Filter, scrape, deduplicate and retrieve the window's documents."""
    allowed = _allowed_documents(session, run, window)
    fetched_now, deferred = _scrape_all(session, allowed)

    dedup_result = dedup.deduplicate(session, sorted(allowed))
    session.commit()

    representatives, new = _unretrieved_representatives(session, run, allowed)
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
    return WindowWork(
        allowed=allowed,
        representatives=representatives,
        retrieval=retrieval,
        fetched_now=fetched_now,
        deferred=deferred,
        copies_marked=dedup_result["newly_marked"],
        new=new,
    )


def _allowed_documents(session: Session, run: Run, window: set[int]) -> set[int]:
    """The window's documents that pass the run's ontology filter."""
    # Filter matches are computed for this window now, not taken from a set
    # computed earlier: a window that finished ingesting after that set was
    # built would otherwise be judged against articles that did not exist yet,
    # find nothing in scope, and be marked done.
    ontology_id = run.ontology_id
    if not ingest_filter.has_links(session, ontology_id):
        return window  # no links at all: the filter cannot distinguish anything
    matches = ingest_filter.matching_documents(
        session, ontology_id, document_ids=sorted(window)
    )
    # Which links fetched this window: live rows change, so record the
    # snapshot on the run.
    snapshot = link_versions.resolve_links(session, ontology_id)
    if snapshot is not None:
        versions.note_filter(run, snapshot.version)
    return window & set(matches)


def _scrape_all(session: Session, allowed: set[int]) -> tuple[int, int]:
    """Scrape every allowed document; the number fetched now, and still deferred."""
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
    return fetched_now, deferred


def _unretrieved_representatives(
    session: Session, run: Run, allowed: set[int]
) -> tuple[set[int], list[Document]]:
    """The window's representatives, and those this run has not retrieved for yet."""
    representatives = set(
        session.scalars(
            select(Document.id).where(
                among(Document.id, allowed),
                Document.body_path.is_not(None),
                Document.is_junk.is_(False),
                Document.duplicate_of.is_(None),
            )
        )
    )
    already = set(
        session.scalars(
            select(Candidate.document_id)
            .where(Candidate.run_id == run.id, among(Candidate.document_id, representatives))
            .distinct()
        )
    )
    new = list(
        session.scalars(select(Document).where(among(Document.id, representatives - already)))
    )
    return representatives, new


def judge_window(
    session: Session, run: Run, representatives: set[int], *, budget: int | None, judge: bool
) -> tuple[int, dict | None]:
    """The selected pairs in the window, and the judge's result if it ran."""
    remaining = None
    if budget is not None:
        judged = count(
            session,
            Verdict.id,
            Verdict.run_id == run.id,
            among(Verdict.document_id, representatives),
        )
        remaining = max(0, budget - judged)
    pairs = count(
        session,
        Candidate.id,
        Candidate.run_id == run.id,
        Candidate.selected.is_(True),
        among(Candidate.document_id, representatives),
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
    return pairs, judged_now


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
                among(Candidate.document_id, representatives),
            )
            .distinct()
        )
    )
    present = set(
        session.scalars(
            select(Candidate.document_id)
            .where(Candidate.run_id == target_run_id, among(Candidate.document_id, retrieved))
            .distinct()
        )
    )
    rows = session.scalars(
        select(Candidate).where(
            Candidate.run_id == source_run_id, among(Candidate.document_id, retrieved - present)
        )
    )
    clone_candidates(session, rows, target_run_id)
    return retrieved
