"""The article quality gate.

This gate is the difference between a corpus of articles and a corpus of 404
pages. Its failures are silent — junk text embeds and prompts perfectly happily —
so the thresholds are pinned here, including the cases where it must *not* fire.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hontology.pipeline.ingest.articles.extract import (
    Extraction,
    extract,
    junk_reason,
    looks_like_article,
)

ARTICLE = (
    "The port authority confirmed on Tuesday that all vessel operations had been "
    "suspended following the storm. Officials said the closure would remain in "
    "effect until at least Thursday morning, affecting more than forty ships "
    "waiting offshore. Container traffic through the terminal handles roughly a "
    "fifth of the region's imports, and the disruption is expected to ripple "
    "through supply chains for several weeks."
)

# Real agency copy is often short. Rejecting it wholesale would bias the corpus
# toward long-form reporting, so the thin check requires BOTH few sentences and
# few words — this passes on word count despite having only two sentences.
WIRE_BRIEF = (
    "A magnitude 6.1 earthquake struck off the eastern coast early on Sunday "
    "afternoon, the national seismological agency reported in a short statement "
    "issued from the capital shortly before midday local time. No casualties were "
    "reported by local authorities in the affected districts and no tsunami warning "
    "was issued for any of the surrounding coastal regions or nearby islands."
)

# Below both thresholds. Two sentences and roughly thirty words is genuinely too
# little to judge a concept against, so the gate rejects it — this pins where the
# boundary actually sits rather than assuming it is generous.
TOO_SHORT_BRIEF = (
    "A magnitude 6.1 earthquake struck off the eastern coast early on Sunday, the "
    "national seismological agency reported. No tsunami warning was issued for the "
    "surrounding region."
)


class TestAccepts:
    def test_a_normal_article(self):
        assert looks_like_article(ARTICLE)
        assert junk_reason(ARTICLE) is None

    def test_a_short_wire_brief_passes_on_word_count(self):
        """Two sentences, but well over the word floor: the `and` is what saves it.

        If the thin check used `or`, this legitimate agency copy would be dropped.
        """
        assert looks_like_article(WIRE_BRIEF)

    def test_an_article_with_some_links(self):
        text = ARTICLE + "\n\nRead more: [our coverage](https://example.test/a)"
        assert looks_like_article(text)


class TestRejects:
    def test_empty(self):
        assert not looks_like_article("")
        assert junk_reason("") == "empty"

    def test_whitespace_only(self):
        assert not looks_like_article("   \n\t  ")
        assert junk_reason("   \n\t  ") == "empty"

    def test_reader_proxy_error_returned_as_200(self):
        text = "Warning: Target URL returned error 404: Not Found\n\n" + ARTICLE
        assert not looks_like_article(text)
        assert junk_reason(text) == "proxy_error"

    @pytest.mark.parametrize(
        "phrase",
        [
            "The page you're looking for cannot be found.",
            "Error 404 - the requested document is unavailable.",
            "Access denied. You do not have permission.",
            "Are you a robot? Please complete the challenge.",
            "Please enable JavaScript to continue.",
            "Subscribe to read the full story.",
            "This website is unavailable in your location. Error 451.",
        ],
    )
    def test_dead_and_wall_pages(self, phrase):
        """These fetch with a 200 and read like text, which is what makes them costly."""
        text = phrase + " " + ARTICLE
        assert not looks_like_article(text)
        assert junk_reason(text) == "dead_page"

    def test_navigation_dump(self):
        nav = " ".join(f"[Section {i}](https://example.test/{i})" for i in range(40))
        assert not looks_like_article(nav)
        assert junk_reason(nav) == "link_dump"

    def test_thin_stub(self):
        text = "Breaking news. More soon."
        assert not looks_like_article(text)
        assert junk_reason(text) == "too_thin"

    def test_a_brief_below_both_thresholds(self):
        """Where the boundary actually is: under 3 sentences AND under 50 words."""
        assert not looks_like_article(TOO_SHORT_BRIEF)
        assert junk_reason(TOO_SHORT_BRIEF) == "too_thin"


class TestChain:
    def test_junk_from_one_extractor_falls_through_to_the_next(self, monkeypatch):
        """A rejected result must not end the chain — that is the whole design."""
        calls: list[str] = []

        def bad(url: str, html: str) -> str:
            calls.append("bad")
            return "The page you're looking for cannot be found."

        def good(url: str, html: str) -> str:
            calls.append("good")
            return ARTICLE

        monkeypatch.setattr(
            "hontology.pipeline.ingest.articles.extract.HTML_EXTRACTORS",
            [("bad", bad), ("good", good)],
        )
        result = extract("https://example.test/a", "<html></html>")

        assert calls == ["bad", "good"]
        assert result.method == "good"
        assert result.text == ARTICLE

    def test_an_extractor_raising_falls_through(self, monkeypatch):
        def boom(url: str, html: str) -> str:
            raise ValueError("no content")

        def good(url: str, html: str) -> str:
            return ARTICLE

        monkeypatch.setattr(
            "hontology.pipeline.ingest.articles.extract.HTML_EXTRACTORS",
            [("boom", boom), ("good", good)],
        )
        assert extract("https://example.test/a", "<html></html>").method == "good"

    def test_all_failing_reports_every_reason(self, monkeypatch):
        def dead(url: str, html: str) -> str:
            return "Error 404 - not found"

        def boom(url: str, html: str) -> str:
            raise ValueError("nope")

        monkeypatch.setattr(
            "hontology.pipeline.ingest.articles.extract.HTML_EXTRACTORS",
            [("dead", dead), ("boom", boom)],
        )
        result = extract("https://example.test/a", "<html></html>")

        assert not result.ok
        assert "dead: rejected (dead_page)" in result.error
        assert "boom: ValueError" in result.error

    def test_extractors_never_run_concurrently(self, monkeypatch):
        """The scraper calls extract() from a thread pool, and trafilatura's
        parser state is shared native memory: overlapping calls crash the
        process rather than raising. At most one extractor may be inside at once.
        """
        inside = 0
        peak = 0
        counter = threading.Lock()

        def slow(url: str, html: str) -> str:
            nonlocal inside, peak
            with counter:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.01)
            with counter:
                inside -= 1
            return ARTICLE

        monkeypatch.setattr(
            "hontology.pipeline.ingest.articles.extract.HTML_EXTRACTORS", [("slow", slow)]
        )
        urls = [f"https://example.test/{i}" for i in range(32)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda url: extract(url, "<html></html>"), urls))

        assert all(r.method == "slow" for r in results)
        assert peak == 1

    def test_no_html_and_no_proxy_is_a_clean_failure(self):
        result = extract("https://example.test/a", None)
        assert not result.ok
        assert result.method == ""

    def test_reader_proxy_is_off_by_default(self, monkeypatch):
        called = False

        def proxy(*args, **kwargs):
            nonlocal called
            called = True
            return ARTICLE

        monkeypatch.setattr(
            "hontology.pipeline.ingest.articles.extract._extract_reader_proxy", proxy
        )
        monkeypatch.setattr("hontology.pipeline.ingest.articles.extract.HTML_EXTRACTORS", [])
        extract("https://example.test/a", "<html></html>")
        assert called is False, "the third-party proxy must be opt-in"


class TestExtraction:
    def test_ok_reflects_text_presence(self):
        assert Extraction("u", "trafilatura", "text").ok
        assert not Extraction("u", "", "", "boom").ok
