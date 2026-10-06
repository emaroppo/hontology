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
import urllib.robotparser
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from urllib.parse import urlsplit

import httpx
from sqlalchemy import Engine, create_engine, func, not_, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.models import Document
from hontology.ingest import extract
from hontology.ingest import filter as filter_module

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


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


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


class RobotsCache:
    """Per-host robots.txt, fetched once per process.

    The shared lock guards only the dictionaries, never the network: a fetch
    holds just its own host's lock. Holding one lock across every fetch made
    the whole thread pool wait in single file on each new host's robots.txt,
    up to its timeout, which is most of a batch when the hosts are new.
    """

    def __init__(self, user_agent: str, *, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._host_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _load(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = httpx.get(
                robots_url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        except httpx.HTTPError:
            # Unreachable: treat as no rules rather than blocking the whole host
            # on a transient network fault.
            return None

        if response.status_code >= 500:
            # RFC 9309: server errors mean "disallow", not "assume allowed". A
            # struggling server is the worst moment to guess in our own favour.
            parser.parse(["User-agent: *", "Disallow: /"])
            return parser
        if response.status_code >= 400:
            # No robots.txt published: everything is allowed.
            return None

        parser.parse(response.text.splitlines())
        return parser

    def allowed(self, url: str) -> bool:
        host = host_of(url)
        with self._lock:
            host_lock = self._host_locks.setdefault(host, threading.Lock())
        with host_lock:
            with self._lock:
                loaded = host in self._parsers
            if not loaded:
                parsed = self._load(url)
                with self._lock:
                    self._parsers[host] = parsed
        with self._lock:
            parser = self._parsers[host]
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        host = host_of(url)
        with self._lock:
            parser = self._parsers.get(host)
        if parser is None:
            return None
        try:
            delay = parser.crawl_delay(self.user_agent)
        except AttributeError:
            return None
        return float(delay) if delay else None


_ROBOTS: dict[str, RobotsCache] = {}
_ROBOTS_LOCK = threading.Lock()


def robots_cache(user_agent: str) -> RobotsCache:
    """The process's robots cache, shared by every scrape batch.

    A batched scrape calls the scraper many times; a cache per call fetched
    the same hosts' robots.txt again on every batch.
    """
    with _ROBOTS_LOCK:
        if user_agent not in _ROBOTS:
            _ROBOTS[user_agent] = RobotsCache(user_agent)
        return _ROBOTS[user_agent]


# The longest a worker sleeps for one host's crawl delay. Beyond it the host's
# remaining documents are deferred to a later batch: a site asking for ten
# minutes between requests is honoured, but no longer holds a whole batch open
# while every other host waits on it.
MAX_POLITE_WAIT_S = 15.0

_LIMITERS: dict[float, HostLimiter] = {}


def host_limiter(default_delay: float) -> HostLimiter:
    """The process's limiter, so a later batch still knows when a host was hit."""
    with _ROBOTS_LOCK:
        if default_delay not in _LIMITERS:
            _LIMITERS[default_delay] = HostLimiter(default_delay)
        return _LIMITERS[default_delay]


class HostLimiter:
    """Enforces a minimum gap between requests to the same host."""

    def __init__(self, default_delay: float) -> None:
        self.default_delay = default_delay
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def ready_in(self, host: str, delay: float | None = None) -> float:
        """Seconds until *host* may be asked again, without reserving the slot."""
        gap = delay if delay is not None else self.default_delay
        with self._lock:
            previous = self._last.get(host)
        if previous is None:
            return 0.0
        return max(0.0, previous + gap - time.monotonic())

    def wait(self, host: str, delay: float | None = None) -> None:
        gap = delay if delay is not None else self.default_delay
        with self._lock:
            previous = self._last.get(host, 0.0)
            now = time.monotonic()
            sleep_for = max(0.0, previous + gap - now)
            self._last[host] = now + sleep_for
        if sleep_for > 0:
            time.sleep(sleep_for)


# Far beyond any article page; a response this large is a feed, an archive or
# a misbehaving server, and reading all of it would only slow the batch.
MAX_PAGE_BYTES = 5 * 1024 * 1024


class _TooSlow(Exception):
    pass


def _get_bounded(url: str, *, user_agent: str, timeout: float) -> tuple[str, int]:
    """GET a page within a total time and size budget.

    httpx's timeout bounds each network operation, not the whole response: a
    server that trickles a byte every few seconds never trips it, and one such
    page held an entire scrape batch open for ten minutes. Streaming the body
    against a deadline makes *timeout* a limit on the whole download.
    """
    deadline = time.monotonic() + timeout
    with httpx.stream(
        "GET",
        url,
        timeout=timeout,
        headers={"User-Agent": user_agent, "Accept": "text/html,*/*"},
        follow_redirects=True,
    ) as response:
        if response.status_code >= 400:
            return "", response.status_code
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_PAGE_BYTES:
                raise _TooSlow(f"page larger than {MAX_PAGE_BYTES} bytes")
            if time.monotonic() > deadline:
                raise _TooSlow(f"download exceeded {timeout:.0f}s")
        body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        return body, response.status_code


def fetch_html(
    url: str, *, user_agent: str, timeout: float, retries: int = 2
) -> tuple[str | None, int | None, str]:
    """Fetch a page. Returns ``(html, status_code, error)``.

    Retries only transient statuses: a 404 or a 403 will not become a 200 by
    asking again, and retrying them just wastes the host's time and ours. A
    page too slow or too large is not retried either: it would be again.
    """
    last_error = ""
    for attempt in range(retries + 1):
        try:
            text, status_code = _get_bounded(url, user_agent=user_agent, timeout=timeout)
        except _TooSlow as exc:
            return None, None, f"TooSlow: {exc}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2.0**attempt)
                continue
            return None, None, last_error

        if status_code in RETRY_STATUS and attempt < retries:
            time.sleep(2.0**attempt)
            last_error = f"HTTP {status_code}"
            continue
        if status_code >= 400:
            return None, status_code, f"HTTP {status_code}"
        return text, status_code, ""

    return None, None, last_error


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
    query = select(func.count(Document.id)).where(
        Document.body_path.is_(None) if retry_failed else Document.fetched_at.is_(None)
    )
    if document_ids is not None:
        query = query.where(among(Document.id, document_ids))
    return session.scalar(query) or 0


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
        documents = pending_documents(
            session,
            budget,
            retry_failed=retry_failed,
            document_ids=sorted(allowed),
            claim=True,
            exclude=exclude,
        )
        filtered_out = _count_pending(
            session, retry_failed=retry_failed, document_ids=document_ids
        ) - _count_pending(session, retry_failed=retry_failed, document_ids=sorted(allowed))
    else:
        documents = pending_documents(
            session,
            budget,
            retry_failed=retry_failed,
            document_ids=document_ids,
            claim=True,
            exclude=exclude,
        )

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
