"""Polite, bounded article fetching.

A feed slice hands us a few hundred URLs across a long tail of hosts. Fetching
them naively — unbounded concurrency, no robots check, no per-host spacing — is
how a research project turns into someone else's incident. Three rules keep this
well-behaved:

- **robots.txt is honored**, cached per host. Following RFC 9309, a 4xx response
  means "no rules, allowed"; a 5xx means "disallowed" rather than "assume yes",
  because a struggling server is the worst time to guess in our own favour.
- **Requests to one host are serialized** with a minimum gap between them.
  Concurrency happens *across* hosts, never within one, so parallelism never
  concentrates on a single site.
- **Every run has a hard budget.** A stray backfill cannot turn into an
  unbounded crawl.

Failures are cached as thoroughly as successes. A dead or paywalled URL with no
row gets retried on every future run forever; recording it means a permanent
failure costs one request ever, with an explicit opt-in to try again.

One host's work is in `pipeline.ingest.articles.host_scrape`; claiming documents
and hosts against other scrapers is in `pipeline.ingest.articles.claims`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.models import Document
from hontology.pipeline.ingest.articles import filter as filter_module
from hontology.pipeline.ingest.articles.claims import (
    count_pending,
    host_lock,
    pending_documents,
)
from hontology.pipeline.ingest.articles.fetch import host_limiter, host_of, robots_cache
from hontology.pipeline.ingest.articles.host_scrape import HostScrape, ScrapeStats

log = logging.getLogger(__name__)


def wait_for_claimed(session: Session, document_ids: list[int]) -> bool:
    """Wait out another scraper's claim on any of these pending documents.

    Returns whether there was one. The caller then looks again: the other
    scraper has fetched them, or left them pending for a later batch.
    """
    pending = select(Document.id).where(
        among(Document.id, document_ids), Document.fetched_at.is_(None)
    )
    free = set(session.scalars(pending.with_for_update(skip_locked=True, of=Document)))
    session.rollback()
    held = set(session.scalars(pending)) - free
    if not held:
        return False
    # Blocks until the other scraper commits its batch.
    session.scalars(
        select(Document.id).where(among(Document.id, sorted(held))).with_for_update(of=Document)
    ).all()
    session.rollback()
    return True


# How long a document deferred by a crawl delay, or left pending after a
# connection failure, is kept out of the next batches. Crawl delays seen in the
# wild reach ten minutes.
HOLD_BACK_S = 600.0
TOTAL_KEYS = (
    "attempted",
    "ok",
    "junk",
    "failed",
    "blocked_by_robots",
    "deferred",
    "retry_later",
)


def drain(
    scrape_batch: Callable[[list[int]], dict],
    *,
    budget: int,
    on_batch: Callable[[dict[str, int]], None] | None = None,
    hold_back_s: float = HOLD_BACK_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Scrape batch after batch until nothing is left or the budget is spent.

    *scrape_batch* takes the ids to hold back and scrapes one batch. A document
    deferred for its host's crawl delay, or left pending after a connection
    failure, is held back for *hold_back_s*: still pending and lowest in id, the
    same documents would otherwise fill every batch, and a batch of nothing but
    them would look like the end of the work. Scraping stops only when a batch
    attempts nothing while nothing is held back; when only held-back documents
    remain, it waits for the first to come due.
    """
    held: dict[int, float] = {}
    totals = dict.fromkeys(TOTAL_KEYS, 0)
    while budget > 0:
        now = clock()
        held = {d: due for d, due in held.items() if due > now}
        result = scrape_batch(sorted(held))
        for document_id in result.get("held_back", []):
            held[document_id] = now + hold_back_s
        for key in TOTAL_KEYS:
            totals[key] += result.get(key, 0)
        if result["attempted"]:
            budget -= result["attempted"]
            if on_batch is not None:
                on_batch(totals)
            continue
        if not held:
            break
        sleep(max(0.0, min(held.values()) - clock()))
    return totals


def _to_fetch(
    session: Session,
    budget: int,
    *,
    retry_failed: bool,
    document_ids: list[int] | None,
    ontology_id: int | None,
    exclude: list[int] | None,
) -> tuple[list[Document], int, dict | None]:
    """The documents to fetch, claimed; how many the code filter skipped; and
    the result to return at once when the filter matched nothing."""
    if ontology_id is None:
        documents = pending_documents(
            session,
            budget,
            retry_failed=retry_failed,
            document_ids=document_ids,
            claim=True,
            exclude=exclude,
        )
        return documents, 0, None

    matches = filter_module.matching_documents(session, ontology_id)
    if not matches:
        # No links, or nothing matched. Either way, fetching zero documents
        # silently would look like the feed had gone quiet.
        return (
            [],
            0,
            ScrapeStats().as_dict()
            | {
                "pending_remaining": len(
                    pending_documents(session, 10_000, retry_failed=retry_failed)
                ),
                "filter": {
                    "ontology_id": ontology_id,
                    "matched_documents": 0,
                    "note": "no documents matched this ontology's code links; "
                    "nothing was fetched",
                },
            },
        )
    # Intersect the filter with any explicit id list.
    allowed = set(matches)
    if document_ids is not None:
        allowed &= set(document_ids)
    # Select among the allowed ids directly. Taking the oldest pending rows
    # first and intersecting afterwards starves the budget whenever the
    # backlog is larger than the window it happens to read.
    wanted = sorted(allowed)
    documents = pending_documents(
        session,
        budget,
        retry_failed=retry_failed,
        document_ids=wanted,
        claim=True,
        exclude=exclude,
    )
    filtered_out = count_pending(
        session, retry_failed=retry_failed, document_ids=document_ids
    ) - count_pending(session, retry_failed=retry_failed, document_ids=wanted)
    return documents, filtered_out, None


def scrape_pending(
    session: Session,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    use_reader_proxy: bool = False,
    document_ids: list[int] | None = None,
    ontology_id: int | None = None,
    exclude: list[int] | None = None,
) -> dict:
    """Fetch and extract bodies for pending documents.

    Work is grouped by host and hosts run in parallel, so the per-host spacing is
    preserved while the long tail still finishes quickly.

    Passing *ontology_id* applies the code filter (see `pipeline.ingest.articles.filter`): only
    documents whose feed events reach one of that ontology's concepts are
    fetched. It is opt-in because it is meaningless for an ontology with no
    curated code links, where applying it would silently fetch nothing.
    """
    settings = get_settings()
    settings.ensure_dirs()
    body_dir = settings.scrape_cache_dir
    budget = limit if limit is not None else settings.scrape_budget

    documents, filtered_out, early = _to_fetch(
        session,
        budget,
        retry_failed=retry_failed,
        document_ids=document_ids,
        ontology_id=ontology_id,
        exclude=exclude,
    )
    if early is not None:
        return early
    if not documents:
        return ScrapeStats().as_dict() | {"pending_remaining": 0}

    job = HostScrape(
        settings=settings,
        robots=robots_cache(settings.scrape_user_agent),
        limiter=host_limiter(settings.scrape_per_host_delay_s),
        use_reader_proxy=use_reader_proxy,
        body_dir=body_dir,
    )
    by_host: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_host[host_of(document.url)].append(document)

    def handle_host_alone(host: str, docs: list[Document]) -> None:
        with host_lock(host):
            job.scrape_host(host, docs)

    workers = min(settings.scrape_max_concurrency, len(by_host))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda item: handle_host_alone(*item), by_host.items()))

    stats = job.stats
    session.flush()
    remaining = len(
        pending_documents(session, 10_000, retry_failed=retry_failed, document_ids=document_ids)
    )
    result = stats.as_dict() | {"pending_remaining": remaining}
    if ontology_id is not None:
        # Reported so a small `attempted` reads as "the filter did its job"
        # rather than "the scraper is broken".
        result["filter"] = {
            "ontology_id": ontology_id,
            "skipped_no_code_match": filtered_out,
        }
    log.info("scrape: %s", result)
    return result | {"held_back": stats.held_back}
