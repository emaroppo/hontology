"""The HTTP side of scraping: robots.txt, per-host spacing, bounded downloads.

- **robots.txt is honored**, cached per host. Following RFC 9309, a 4xx response
  means "no rules, allowed"; a 5xx means "disallowed" rather than "assume yes",
  because a struggling server is the worst time to guess in our own favour.
- **Requests to one host are spaced** by a minimum gap (`HostLimiter`).
- **A download is bounded** in total time and size, not just per operation.
"""

from __future__ import annotations

import threading
import time
import urllib.robotparser
from urllib.parse import urlsplit

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


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
