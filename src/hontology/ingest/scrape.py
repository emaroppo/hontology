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
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import ColumnElement, Engine, create_engine, not_, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.lookups import count
from hontology.db.models import Document
from hontology.ingest import extract
from hontology.ingest import filter as filter_module
from hontology.ingest.fetch import RobotsCache as RobotsCache
from hontology.ingest.fetch import fetch_html, host_limiter, host_of, robots_cache

log = logging.getLogger(__name__)

# A connection failure (DNS or a refused connection) is tried again in later
# batches, and recorded as a failure only at the last of these attempts.
CONNECT_ERROR = "ConnectError"
MAX_CONNECT_FAILURES = 3


@dataclass
class ScrapeStats:
    attempted: int = 0
    ok: int = 0
    junk: int = 0
    failed: int = 0
    blocked: int = 0
    # Left pending because the host's crawl delay would have stalled the batch.
    deferred: int = 0
    # Left pending after a connection failure, to be tried in a later batch.
    retry_later: int = 0
    # Ids of the documents deferred or left for later, for a caller to hold back.
    held_back: list[int] = field(default_factory=list)
    methods: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "junk": self.junk,
            "failed": self.failed,
            "blocked_by_robots": self.blocked,
            "deferred": self.deferred,
            "retry_later": self.retry_later,
            "methods": dict(self.methods),
        }


# The longest a worker sleeps for one host's crawl delay. Beyond it the host's
# remaining documents are deferred to a later batch: a site asking for ten
# minutes between requests is honoured, but no longer holds a whole batch open
# while every other host waits on it.
MAX_POLITE_WAIT_S = 15.0


def _record(
    document: Document,
    *,
    result: extract.Extraction | None,
    status_code: int | None,
    error: str,
    body_dir,
) -> str:
    """Write the outcome onto the document row, body to disk when we got one."""
    document.fetched_at = datetime.now(UTC)
    document.http_status = status_code
    document.extractor_version = _trafilatura_version()

    if result is not None and result.ok:
        path = body_dir / f"{document.url_hash}.txt"
        path.write_text(result.text, encoding="utf-8")
        document.body_path = path.name  # relative: the cache survives a move
        document.body_chars = len(result.text)
        document.extractor = result.method
        document.error = None
        document.is_junk = False
        document.junk_reason = None
        return "ok"

    document.body_path = None
    document.body_chars = 0
    document.error = (error or (result.error if result else ""))[:1000] or "no text extracted"

    # A page we fetched but rejected is junk; one we could not fetch is a failure.
    if status_code is not None and status_code < 400 and result is not None:
        document.is_junk = True
        document.junk_reason = "gate_rejected"
        return "junk"

    document.is_junk = False
    document.junk_reason = None
    return "failed"


def _trafilatura_version() -> str | None:
    try:
        from importlib.metadata import version

        return f"trafilatura {version('trafilatura')}"
    except Exception:  # noqa: BLE001 - provenance is nice to have, never required
        return None


def pending_documents(
    session: Session,
    limit: int,
    *,
    retry_failed: bool = False,
    document_ids: list[int] | None = None,
    claim: bool = False,
    exclude: list[int] | None = None,
) -> list[Document]:
    """Documents that still need a body.

    By default only never-attempted ones, so a run does not spend its budget
    re-hitting URLs already known to be dead. ``document_ids`` narrows the work
    to a specific set — useful for fetching exactly the documents one run needs
    rather than draining the whole backlog.

    With *claim*, the rows are locked until the caller commits, and rows another
    scraper has claimed are skipped, so scrapers running side by side never
    fetch the same document or double the rate on a host.

    *exclude* leaves out documents a caller is holding back for now, such as
    ones whose host asked for a long crawl delay.
    """
    query = select(Document)
    query = (
        query.where(Document.fetched_at.is_(None))
        if not retry_failed
        else query.where(Document.body_path.is_(None))
    )
    if document_ids is not None:
        query = query.where(among(Document.id, document_ids))
    if exclude:
        query = query.where(not_(among(Document.id, exclude)))
    query = query.order_by(Document.id).limit(limit)
    if claim:
        query = query.with_for_update(skip_locked=True, of=Document)
    return list(session.scalars(query))


@lru_cache
def _lock_engine(url: str) -> Engine:
    # Unpooled: each host holds its own connection only while it is scraped,
    # and a pool sized for the app would make the workers queue for one.
    return create_engine(url, poolclass=NullPool, future=True)


@contextmanager
def host_lock(host: str) -> Iterator[None]:
    """Hold a host for this process, across every scraper on the database.

    The per-host spacing is kept inside one process; a second scraper working
    on other documents from the same host would double the rate it sees. This
    waits until no other scraper is on the host.
    """
    with _lock_engine(get_settings().database_url).connect() as connection:
        key = {"host": host}
        connection.execute(text("SELECT pg_advisory_lock(hashtextextended(:host, 0))"), key)
        try:
            yield
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(hashtextextended(:host, 0))"), key
            )


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


def _count_pending(
    session: Session, *, retry_failed: bool, document_ids: list[int] | None
) -> int:
    where: list[ColumnElement[bool]] = [
        Document.body_path.is_(None) if retry_failed else Document.fetched_at.is_(None)
    ]
    if document_ids is not None:
        where.append(among(Document.id, document_ids))
    return count(session, Document.id, *where)


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

    Passing *ontology_id* applies the code filter (see `ingest.filter`): only
    documents whose feed events reach one of that ontology's concepts are
    fetched. It is opt-in because it is meaningless for an ontology with no
    curated code links, where applying it would silently fetch nothing.
    """
    settings = get_settings()
    settings.ensure_dirs()
    body_dir = settings.scrape_cache_dir
    budget = limit if limit is not None else settings.scrape_budget

    filtered_out = 0
    if ontology_id is not None:
        matches = filter_module.matching_documents(session, ontology_id)
        if not matches:
            # No links, or nothing matched. Either way, fetching zero documents
            # silently would look like the feed had gone quiet.
            return ScrapeStats().as_dict() | {
                "pending_remaining": len(
                    pending_documents(session, 10_000, retry_failed=retry_failed)
                ),
                "filter": {
                    "ontology_id": ontology_id,
                    "matched_documents": 0,
                    "note": "no documents matched this ontology's code links; "
                    "nothing was fetched",
                },
            }
        # Intersect the filter with any explicit id list.
        allowed = set(matches)
        if document_ids is not None:
            allowed &= set(document_ids)
        # Select among the allowed ids directly. Taking the oldest pending rows
        # first and intersecting afterwards starves the budget whenever the
        # backlog is larger than the window it happens to read.
        wanted: list[int] | None = sorted(allowed)
    else:
        wanted = document_ids
    documents = pending_documents(
        session,
        budget,
        retry_failed=retry_failed,
        document_ids=wanted,
        claim=True,
        exclude=exclude,
    )
    if ontology_id is not None:
        filtered_out = _count_pending(
            session, retry_failed=retry_failed, document_ids=document_ids
        ) - _count_pending(session, retry_failed=retry_failed, document_ids=wanted)

    if not documents:
        return ScrapeStats().as_dict() | {"pending_remaining": 0}

    robots = robots_cache(settings.scrape_user_agent)
    limiter = host_limiter(settings.scrape_per_host_delay_s)
    stats = ScrapeStats()
    stats_lock = threading.Lock()

    by_host: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_host[host_of(document.url)].append(document)

    def handle_host(host: str, docs: list[Document]) -> None:
        for document in docs:
            if not robots.allowed(document.url):
                document.fetched_at = datetime.now(UTC)
                document.error = "disallowed by robots.txt"
                document.is_junk = False
                with stats_lock:
                    stats.attempted += 1
                    stats.blocked += 1
                continue

            delay = robots.crawl_delay(document.url)
            if limiter.ready_in(host, delay) > MAX_POLITE_WAIT_S:
                # This and the host's remaining documents wait for a later
                # batch; they stay pending, so nothing is lost.
                with stats_lock:
                    stats.deferred += len(docs) - docs.index(document)
                    stats.held_back.extend(d.id for d in docs[docs.index(document) :])
                return
            limiter.wait(host, delay)
            html, status_code, error = fetch_html(
                document.url,
                user_agent=settings.scrape_user_agent,
                timeout=settings.scrape_timeout_s,
            )
            if error.startswith(CONNECT_ERROR) and (
                document.connect_failures + 1 < MAX_CONNECT_FAILURES
            ):
                # The name did not resolve or the host did not answer. A
                # network blip looks exactly like a dead host, so the document
                # stays pending for a later batch instead of failing for good.
                document.connect_failures += 1
                document.error = error[:1000]
                with stats_lock:
                    stats.retry_later += 1
                    stats.held_back.append(document.id)
                continue
            result = (
                extract.extract(
                    document.url,
                    html,
                    use_reader_proxy=use_reader_proxy,
                    timeout=settings.scrape_timeout_s,
                    user_agent=settings.scrape_user_agent,
                )
                if html or use_reader_proxy
                else None
            )
            outcome = _record(
                document,
                result=result,
                status_code=status_code,
                error=error,
                body_dir=body_dir,
            )
            with stats_lock:
                stats.attempted += 1
                setattr(stats, outcome, getattr(stats, outcome) + 1)
                if result is not None and result.method:
                    stats.methods[result.method] += 1

    workers = min(settings.scrape_max_concurrency, len(by_host))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:

        def handle_host_alone(host: str, docs: list[Document]) -> None:
            with host_lock(host):
                handle_host(host, docs)

        list(pool.map(lambda item: handle_host_alone(*item), by_host.items()))

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
