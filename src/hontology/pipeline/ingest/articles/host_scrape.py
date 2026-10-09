"""Scraping one host's documents: robots, spacing, fetching and recording.

A host's documents are fetched one after another with the host's minimum gap
between them; `pipeline.ingest.articles.scrape` runs hosts side by side. Each
outcome is written onto the document row, failures as thoroughly as successes.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hontology.config import Settings
from hontology.db.models import Document
from hontology.pipeline.ingest.articles import extract
from hontology.pipeline.ingest.articles.fetch import HostLimiter, RobotsCache, fetch_html

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


def record(
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


@dataclass
class HostScrape:
    """What every host's worker shares in one batch."""

    settings: Settings
    robots: RobotsCache
    limiter: HostLimiter
    use_reader_proxy: bool
    body_dir: Path
    stats: ScrapeStats = field(default_factory=ScrapeStats)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def scrape_host(self, host: str, docs: list[Document]) -> None:
        """Fetch one host's documents in turn, honouring robots.txt and its delay."""
        for position, document in enumerate(docs):
            if not self.robots.allowed(document.url):
                document.fetched_at = datetime.now(UTC)
                document.error = "disallowed by robots.txt"
                document.is_junk = False
                with self.lock:
                    self.stats.attempted += 1
                    self.stats.blocked += 1
                continue

            delay = self.robots.crawl_delay(document.url)
            if self.limiter.ready_in(host, delay) > MAX_POLITE_WAIT_S:
                # This and the host's remaining documents wait for a later
                # batch; they stay pending, so nothing is lost.
                with self.lock:
                    self.stats.deferred += len(docs) - position
                    self.stats.held_back.extend(d.id for d in docs[position:])
                return
            self.limiter.wait(host, delay)
            self._fetch(document)

    def _fetch(self, document: Document) -> None:
        settings, use_reader_proxy = self.settings, self.use_reader_proxy
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
            with self.lock:
                self.stats.retry_later += 1
                self.stats.held_back.append(document.id)
            return
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
        outcome = record(
            document,
            result=result,
            status_code=status_code,
            error=error,
            body_dir=self.body_dir,
        )
        with self.lock:
            self.stats.attempted += 1
            setattr(self.stats, outcome, getattr(self.stats, outcome) + 1)
            if result is not None and result.method:
                self.stats.methods[result.method] += 1
