"""Fetch behaviour: robots, retries and per-host spacing.

All mocked — a test suite must not depend on, or hammer, live sites.
"""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from hontology.ingest.scrape import HostLimiter, RobotsCache, fetch_html, host_of

AGENT = "hontology-test"


class TestHostOf:
    def test_extracts_and_lowercases(self):
        assert host_of("https://Example.TEST/a/b?c=1") == "example.test"

    def test_distinguishes_subdomains(self):
        assert host_of("https://a.example.test/x") != host_of("https://b.example.test/x")


class TestRobots:
    @respx.mock
    def test_disallowed_path_is_blocked(self):
        respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        )
        robots = RobotsCache(AGENT)
        assert robots.allowed("https://example.test/public/a") is True
        assert robots.allowed("https://example.test/private/a") is False

    @respx.mock
    def test_missing_robots_allows_everything(self):
        """404 means no rules published, which is permission."""
        respx.get("https://example.test/robots.txt").mock(return_value=httpx.Response(404))
        assert RobotsCache(AGENT).allowed("https://example.test/anything") is True

    @respx.mock
    def test_server_error_disallows(self):
        """RFC 9309: 5xx means disallow, not 'assume we may'.

        A struggling server is the worst moment to guess in our own favour.
        """
        respx.get("https://example.test/robots.txt").mock(return_value=httpx.Response(503))
        assert RobotsCache(AGENT).allowed("https://example.test/anything") is False

    @respx.mock
    def test_unreachable_host_does_not_block_everything(self):
        """A transient network fault must not look like a site-wide ban."""
        respx.get("https://example.test/robots.txt").mock(
            side_effect=httpx.ConnectError("boom")
        )
        assert RobotsCache(AGENT).allowed("https://example.test/anything") is True

    @respx.mock
    def test_robots_is_fetched_once_per_host(self):
        route = respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow:\n")
        )
        robots = RobotsCache(AGENT)
        for i in range(5):
            robots.allowed(f"https://example.test/page/{i}")
        assert route.call_count == 1

    @respx.mock
    def test_crawl_delay_is_read(self):
        respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow:\nCrawl-delay: 7\n")
        )
        robots = RobotsCache(AGENT)
        robots.allowed("https://example.test/a")
        assert robots.crawl_delay("https://example.test/a") == 7.0


class TestFetchRetries:
    @respx.mock
    def test_success_returns_html(self):
        respx.get("https://example.test/a").mock(
            return_value=httpx.Response(200, text="<html>hi</html>")
        )
        html, status, error = fetch_html("https://example.test/a", user_agent=AGENT, timeout=5)
        assert html == "<html>hi</html>"
        assert status == 200
        assert error == ""

    @respx.mock
    def test_404_is_not_retried(self):
        """Asking again will not turn a 404 into a 200; it just wastes requests."""
        route = respx.get("https://example.test/a").mock(return_value=httpx.Response(404))
        html, status, error = fetch_html(
            "https://example.test/a", user_agent=AGENT, timeout=5, retries=2
        )
        assert route.call_count == 1
        assert html is None
        assert status == 404
        assert "404" in error

    @respx.mock
    def test_403_is_not_retried(self):
        route = respx.get("https://example.test/a").mock(return_value=httpx.Response(403))
        fetch_html("https://example.test/a", user_agent=AGENT, timeout=5, retries=2)
        assert route.call_count == 1

    @respx.mock
    def test_transient_500_is_retried_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        route = respx.get("https://example.test/a").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, text="<html>ok</html>"),
            ]
        )
        html, status, _ = fetch_html(
            "https://example.test/a", user_agent=AGENT, timeout=5, retries=2
        )
        assert route.call_count == 2
        assert html == "<html>ok</html>"
        assert status == 200

    @respx.mock
    def test_429_is_retried(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        route = respx.get("https://example.test/a").mock(
            side_effect=[httpx.Response(429), httpx.Response(200, text="<html/>")]
        )
        fetch_html("https://example.test/a", user_agent=AGENT, timeout=5, retries=2)
        assert route.call_count == 2

    @respx.mock
    def test_retries_are_bounded(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        route = respx.get("https://example.test/a").mock(return_value=httpx.Response(503))
        fetch_html("https://example.test/a", user_agent=AGENT, timeout=5, retries=2)
        assert route.call_count == 3  # initial attempt plus two retries

    @respx.mock
    def test_the_user_agent_is_sent(self):
        """Identifying the crawler honestly is the point of having one."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["ua"] = request.headers.get("User-Agent")
            return httpx.Response(200, text="<html/>")

        respx.get("https://example.test/a").mock(side_effect=handler)
        fetch_html("https://example.test/a", user_agent=AGENT, timeout=5)
        assert captured["ua"] == AGENT


class TestHostLimiter:
    def test_first_request_to_a_host_does_not_wait(self):
        limiter = HostLimiter(default_delay=5.0)
        started = time.monotonic()
        limiter.wait("example.test")
        assert time.monotonic() - started < 0.2

    def test_second_request_to_the_same_host_waits(self):
        limiter = HostLimiter(default_delay=0.25)
        limiter.wait("example.test")
        started = time.monotonic()
        limiter.wait("example.test")
        assert time.monotonic() - started >= 0.2

    def test_different_hosts_do_not_block_each_other(self):
        """Concurrency across hosts is the whole point; only one host is serialized."""
        limiter = HostLimiter(default_delay=0.25)
        limiter.wait("a.test")
        started = time.monotonic()
        limiter.wait("b.test")
        assert time.monotonic() - started < 0.2

    def test_an_explicit_crawl_delay_overrides_the_default(self):
        limiter = HostLimiter(default_delay=0.0)
        limiter.wait("example.test", 0.25)
        started = time.monotonic()
        limiter.wait("example.test", 0.25)
        assert time.monotonic() - started >= 0.2


@pytest.mark.requires_db
class TestScrapePending:
    def test_a_blocked_document_is_recorded_not_silently_skipped(self):
        """A robots block must be visible on the row, not an invisible no-op."""
        from sqlalchemy import text as sql_text

        from hontology.db.models import Document
        from hontology.db.session import session_scope
        from hontology.ingest.scrape import scrape_pending

        url = "https://blocked.test/article"
        with session_scope() as session:
            session.execute(sql_text("DELETE FROM documents WHERE url = :u"), {"u": url})
            session.add(Document(url=url, url_hash="blockedtesthash1"))

        with respx.mock:
            respx.get("https://blocked.test/robots.txt").mock(
                return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
            )
            with session_scope() as session:
                doc = session.query(Document).filter_by(url=url).one()
                # Scoped to this document: the dev database holds a real backlog.
                result = scrape_pending(session, limit=1, document_ids=[doc.id])
                assert result["blocked_by_robots"] == 1
                session.refresh(doc)
                assert doc.fetched_at is not None
                assert "robots" in (doc.error or "")

        with session_scope() as session:
            session.execute(sql_text("DELETE FROM documents WHERE url = :u"), {"u": url})
