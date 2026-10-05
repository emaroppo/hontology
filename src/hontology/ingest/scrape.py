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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Document
from hontology.ingest import extract
from hontology.ingest import filter as filter_module

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class ScrapeStats:
    attempted: int = 0
    ok: int = 0
    junk: int = 0
    failed: int = 0
    blocked: int = 0
    methods: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "junk": self.junk,
            "failed": self.failed,
            "blocked_by_robots": self.blocked,
            "methods": dict(self.methods),
        }


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


class RobotsCache:
    """Per-host robots.txt, fetched once per process."""

    def __init__(self, user_agent: str, *, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
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
            if host not in self._parsers:
                self._parsers[host] = self._load(url)
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


class HostLimiter:
    """Enforces a minimum gap between requests to the same host."""

    def __init__(self, default_delay: float) -> None:
        self.default_delay = default_delay
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, delay: float | None = None) -> None:
        gap = delay if delay is not None else self.default_delay
        with self._lock:
            previous = self._last.get(host, 0.0)
            now = time.monotonic()
            sleep_for = max(0.0, previous + gap - now)
            self._last[host] = now + sleep_for
        if sleep_for > 0:
            time.sleep(sleep_for)


def fetch_html(
    url: str, *, user_agent: str, timeout: float, retries: int = 2
) -> tuple[str | None, int | None, str]:
    """Fetch a page. Returns ``(html, status_code, error)``.

    Retries only transient statuses: a 404 or a 403 will not become a 200 by
    asking again, and retrying them just wastes the host's time and ours.
    """
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = httpx.get(
                url,
                timeout=timeout,
                headers={"User-Agent": user_agent, "Accept": "text/html,*/*"},
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2.0**attempt)
                continue
            return None, None, last_error

        if response.status_code in RETRY_STATUS and attempt < retries:
            time.sleep(2.0**attempt)
            last_error = f"HTTP {response.status_code}"
            continue
        if response.status_code >= 400:
            return None, response.status_code, f"HTTP {response.status_code}"
        return response.text, response.status_code, ""

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
) -> list[Document]:
    """Documents that still need a body.

    By default only never-attempted ones, so a run does not spend its budget
    re-hitting URLs already known to be dead. ``document_ids`` narrows the work
    to a specific set — useful for fetching exactly the documents one run needs
    rather than draining the whole backlog.
    """
    query = select(Document)
    query = (
        query.where(Document.fetched_at.is_(None))
        if not retry_failed
        else query.where(Document.body_path.is_(None))
    )
    if document_ids is not None:
        query = query.where(Document.id.in_(document_ids))
    return list(session.scalars(query.order_by(Document.id).limit(limit)))


def _count_pending(
    session: Session, *, retry_failed: bool, document_ids: list[int] | None
) -> int:
    query = select(func.count(Document.id)).where(
        Document.body_path.is_(None) if retry_failed else Document.fetched_at.is_(None)
    )
    if document_ids is not None:
        query = query.where(Document.id.in_(document_ids))
    return session.scalar(query) or 0


def scrape_pending(
    session: Session,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    use_reader_proxy: bool = False,
    document_ids: list[int] | None = None,
    ontology_id: int | None = None,
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
            session, budget, retry_failed=retry_failed, document_ids=sorted(allowed)
        )
        filtered_out = _count_pending(
            session, retry_failed=retry_failed, document_ids=document_ids
        ) - _count_pending(session, retry_failed=retry_failed, document_ids=sorted(allowed))
    else:
        documents = pending_documents(
            session, budget, retry_failed=retry_failed, document_ids=document_ids
        )

    if not documents:
        return ScrapeStats().as_dict() | {"pending_remaining": 0}

    robots = RobotsCache(settings.scrape_user_agent)
    limiter = HostLimiter(settings.scrape_per_host_delay_s)
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

            limiter.wait(host, robots.crawl_delay(document.url))
            html, status_code, error = fetch_html(
                document.url,
                user_agent=settings.scrape_user_agent,
                timeout=settings.scrape_timeout_s,
            )
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
        list(pool.map(lambda item: handle_host(*item), by_host.items()))

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
    return result
