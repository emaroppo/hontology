"""Article text extraction: a fallback chain behind a quality gate.

No single extractor handles the whole open web. Each is tried in turn and the
first result that passes the gate wins.

**The gate is the important part.** A failed fetch is easy to detect; the
expensive failure is a *successful* fetch of something that is not an article —
a 404 page rendered with a 200, a paywall stub, a cookie wall, a navigation dump.
Those produce plausible text that flows all the way into embeddings and judge
prompts and quietly degrades everything downstream.

So junk is treated as **failure**, not as a result: it does not stop the chain,
it falls through to the next extractor. That single decision is what makes the
chain worth having, because extractors fail on different pages in different ways.

The thin-content check requires *both* few sentences and few words, which
deliberately spares short wire briefs: agency copy is often a single dense
paragraph, and rejecting it would bias the corpus toward long-form reporting.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import httpx

# A reader proxy answers 200 even when it could not load the target, and says so
# on the first line of the body instead.
_PROXY_ERROR = re.compile(r"^Warning: Target URL returned error \d+", re.M)

# Pages that fetch successfully but are not the article, grouped by what went
# wrong. Each entry is one regex alternative; they are joined into a single
# case-insensitive pattern.
_DEAD_PAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "missing": (
        r"page (?:not found|(?:cannot|could not|can't) be found|has (?:been )?moved)",
        r"looking for (?:cannot|could not|can't) be found",
        r"\b404\b[^\n]{0,80}\berror|\berror\s*(?:code\s*)?404\b",
    ),
    "bot_wall": (
        r"access denied",
        r"are you a (?:human|robot)",
        r"verify (?:that )?you are",
        r"(?:enable|turn on) javascript",
    ),
    "paywall": (r"subscribe (?:now )?to (?:read|continue)",),
    "geo_block": (r"not available in your (?:country|region|location)",),
}
_DEAD_PAGE = re.compile(
    "|".join(p for group in _DEAD_PAGE_MARKERS.values() for p in group), re.I
)

_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
# A sentence, for counting purposes: starts with a letter, runs at least 30
# characters, ends in terminal punctuation. Shorter fragments are menu items and
# captions more often than prose.
_SENTENCE = re.compile(r"[A-Za-z][^.!?]{29,}[.!?]")

# Above this share of characters sitting inside links, the text is navigation
# chrome rather than prose.
_MAX_LINK_DENSITY = 0.47
_MIN_SENTENCES = 3
_MIN_WORDS = 50


@dataclass(frozen=True)
class Extraction:
    url: str
    method: str
    text: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text)


def junk_reason(text: str) -> str | None:
    """Why the gate rejects *text*, or None if it reads like an article.

    Also the value of the ``junk_reason`` column. See the module docstring for
    why each threshold is where it is.
    """
    if not text or not text.strip():
        return "empty"
    if _PROXY_ERROR.search(text):
        return "proxy_error"
    if _DEAD_PAGE.search(text):
        return "dead_page"

    link_chars = sum(m.end() - m.start() for m in _LINK.finditer(text))
    if link_chars > _MAX_LINK_DENSITY * len(text):
        return "link_dump"

    prose = _LINK.sub(" ", text)
    thin = len(_SENTENCE.findall(prose)) < _MIN_SENTENCES
    short = len(prose.split()) < _MIN_WORDS
    return "too_thin" if thin and short else None


def looks_like_article(text: str) -> bool:
    """Whether *text* passes the gate: not dead, not navigation, not a stub."""
    return junk_reason(text) is None


# ---------------------------------------------------------------------------
# Extractors
#
# Each takes already-fetched HTML where possible, so the chain costs one network
# request rather than one per method. Only the reader proxy needs its own call,
# which is part of why it is last and opt-in.
# ---------------------------------------------------------------------------


def _extract_trafilatura(url: str, html: str) -> str:
    import trafilatura

    text = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,  # drop borderline boilerplate rather than keep it
        include_comments=False,
        include_tables=False,
        deduplicate=True,
    )
    if not text or not text.strip():
        raise ValueError("empty text")
    return text.strip()


def _extract_readability(url: str, html: str) -> str:
    from bs4 import BeautifulSoup
    from readability import Document

    summary = Document(html).summary()
    text = BeautifulSoup(summary, "html.parser").get_text("\n").strip()
    if not text:
        raise ValueError("empty text")
    return text


# (name, function) in priority order.
HTML_EXTRACTORS: list[tuple[str, Callable[[str, str], str]]] = [
    ("trafilatura", _extract_trafilatura),
    ("readability", _extract_readability),
]

READER_PROXY_ENDPOINT = "https://r.jina.ai/"


def _extract_reader_proxy(url: str, *, timeout: float, user_agent: str) -> str:
    """Last resort: a third-party reader service renders the page for us.

    Off by default. It sends the target URL to an external host, which is a
    different privacy and dependency posture from fetching the page ourselves,
    so it is an explicit opt-in rather than a silent fallback.
    """
    response = httpx.get(
        READER_PROXY_ENDPOINT + url,
        timeout=timeout,
        headers={"Accept": "text/plain", "User-Agent": user_agent},
        follow_redirects=True,
    )
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        raise ValueError("empty text")
    return text


def extract(
    url: str,
    html: str | None,
    *,
    use_reader_proxy: bool = False,
    timeout: float = 20.0,
    user_agent: str = "hontology",
) -> Extraction:
    """Run the chain, returning the first extraction that passes the gate."""
    errors: list[str] = []

    if html:
        for name, extractor in HTML_EXTRACTORS:
            try:
                text = extractor(url, html)
            except ImportError as exc:
                errors.append(f"{name}: not installed ({exc.name})")
                continue
            except Exception as exc:  # noqa: BLE001 - any failure just tries the next
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if not looks_like_article(text):
                # Junk is a failure, so the chain keeps going.
                errors.append(f"{name}: rejected ({junk_reason(text)})")
                continue
            return Extraction(url=url, method=name, text=text)

    if use_reader_proxy:
        try:
            text = _extract_reader_proxy(url, timeout=timeout, user_agent=user_agent)
            if looks_like_article(text):
                return Extraction(url=url, method="reader_proxy", text=text)
            errors.append(f"reader_proxy: rejected ({junk_reason(text)})")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reader_proxy: {type(exc).__name__}: {exc}")

    return Extraction(url=url, method="", text="", error=" | ".join(errors))
