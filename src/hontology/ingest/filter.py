"""Deciding which documents are worth fetching.

The feed tags every event with a CAMEO code before anyone has fetched anything.
That is the one signal available *pre-scrape*, and it is why this lives at the
ingest boundary rather than in retrieval: semantic similarity needs the article
body to compare against, so it cannot tell you whether the body was worth
downloading in the first place.

Scraping is the expensive stage — a network request per article, rate-limited per
host, against a feed producing a few hundred new URLs every fifteen minutes. A
filter here spends that budget on documents an ontology could plausibly match
instead of on whatever happened to arrive first.

**The filter is deliberately optional and off by default.** It only means
anything for an ontology whose concepts have been mapped onto the code system; a
supply-chain or corporate-events ontology maps onto nothing, and silently
applying the filter would starve its corpus to zero.

**Two feeds, one set of links.** CAMEO links match the event export's coded
events; GKG theme links match the knowledge graph's article themes. CAMEO only
sees articles from which a political event could be coded, so families with no
actors (fires, cyberattacks, shortages) reach the filter through themes. A
document passes if either feed matches it.

**Everything is still ingested.** Only fetching is gated. Feed rows are cheap and
the corpus is shared between ontologies, so discarding a document because *this*
ontology cannot use it would corrupt the corpus for the next one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import (
    Code,
    CodeSystem,
    Concept,
    ConceptCode,
    Document,
    FeedArticle,
    FeedEvent,
)
from hontology.ingest.cameo import CAMEO_SLUG
from hontology.ingest.themes import THEMES_LEVEL, THEMES_SLUG

# Most specific first.
CODE_TIERS = ("event_code", "base_code", "root_code")


@dataclass(frozen=True)
class Match:
    """Why a document survived the filter."""

    document_id: int
    code: str
    level: str
    concept_ids: frozenset[int]


def concept_code_map(
    session: Session, ontology_id: int, *, system: str = CAMEO_SLUG
) -> dict[str, set[int]]:
    """``{code string: concepts linked to it}`` for one ontology and code system.

    Empty when the ontology has no curated links, which callers must treat as
    "cannot filter" rather than "nothing matches".
    """
    concept_ids = {
        c.id for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    if not concept_ids:
        return {}

    links: dict[str, set[int]] = {}
    for link, code in session.execute(
        select(ConceptCode, Code)
        .join(Code, Code.id == ConceptCode.code_id)
        .join(CodeSystem, CodeSystem.id == Code.system_id)
        .where(ConceptCode.concept_id.in_(concept_ids), CodeSystem.slug == system)
    ).all():
        links.setdefault(code.code, set()).add(link.concept_id)
    return links


def concept_theme_map(session: Session, ontology_id: int) -> dict[str, set[int]]:
    """``{GKG theme: concepts linked to it}`` for one ontology."""
    return concept_code_map(session, ontology_id, system=THEMES_SLUG)


def resolve_event(event: Any, links: dict[str, set[int]]) -> tuple[str, str, set[int]] | None:
    """Resolve one feed event to ``(code, level, concepts)``, or None.

    The fallback walks event → base → root and stops at the **first tier that
    carries a code**, whether or not that code matched. Falling through to a
    broader tier after a specific code failed would pull in far more concepts
    than the event actually supports: an event coded 1451 ("riot") that matches
    nothing should not be re-tried as 14 ("PROTEST") and match everything
    protest-shaped.
    """
    for tier in CODE_TIERS:
        code_value = getattr(event, tier)
        if not code_value:
            continue
        hits = links.get(code_value)
        # This tier had a code, so it decides — matched or not.
        return (code_value, tier.removesuffix("_code"), set(hits)) if hits else None
    return None


def matching_documents(
    session: Session,
    ontology_id: int,
    *,
    document_ids: list[int] | None = None,
) -> dict[int, Match]:
    """Documents whose feed records reach at least one concept in this ontology.

    CAMEO matches come first, so a document both feeds reach reports its event
    code; the theme pass adds documents only the knowledge graph saw.
    """
    links = concept_code_map(session, ontology_id)
    themes = concept_theme_map(session, ontology_id)
    if not links and not themes:
        return {}

    # Only the columns the match needs, streamed: over the whole corpus this is
    # millions of rows, and loading them as objects exhausted memory.
    query = select(
        FeedEvent.document_id, FeedEvent.event_code, FeedEvent.base_code, FeedEvent.root_code
    ).where(FeedEvent.document_id.is_not(None))
    if document_ids is not None:
        query = query.where(among(FeedEvent.document_id, document_ids))

    matches: dict[int, Match] = {}
    for event in session.execute(query.execution_options(yield_per=50_000)):
        if event.document_id is None or event.document_id in matches:
            continue
        resolved = resolve_event(event, links)
        if resolved is None:
            continue
        code, level, concept_ids = resolved
        matches[event.document_id] = Match(
            document_id=event.document_id,
            code=code,
            level=level,
            concept_ids=frozenset(concept_ids),
        )

    if themes:
        theme_query = select(FeedArticle.document_id, FeedArticle.themes).where(
            FeedArticle.document_id.is_not(None),
            FeedArticle.themes.overlap(sorted(themes)),
        )
        if document_ids is not None:
            theme_query = theme_query.where(among(FeedArticle.document_id, document_ids))
        for document_id, article_themes in session.execute(theme_query):
            if document_id is None or document_id in matches:
                continue
            hit = sorted(t for t in article_themes if t in themes)
            matches[document_id] = Match(
                document_id=document_id,
                code=hit[0],
                level=THEMES_LEVEL,
                concept_ids=frozenset().union(*(themes[t] for t in hit)),
            )
    return matches


def preview(session: Session, ontology_id: int) -> dict:
    """What the filter would do, without doing it.

    Reports the unfetched share separately because that is the number the scrape
    budget is actually spent against.
    """
    links = concept_code_map(session, ontology_id) | concept_theme_map(session, ontology_id)
    total = len(list(session.scalars(select(Document.id))))
    unfetched = [
        row for row in session.scalars(select(Document.id).where(Document.fetched_at.is_(None)))
    ]

    if not links:
        return {
            "usable": False,
            "reason": (
                "this ontology has no concept↔code links, so the filter cannot "
                "distinguish anything. Curate links on the Code Links page, or "
                "leave the filter off."
            ),
            "linked_codes": 0,
            "documents_total": total,
            "documents_unfetched": len(unfetched),
            "documents_matching": None,
            "unfetched_matching": None,
        }

    all_matches = matching_documents(session, ontology_id)
    unfetched_matching = sum(1 for doc_id in unfetched if doc_id in all_matches)

    return {
        "usable": True,
        "linked_codes": len(links),
        "documents_total": total,
        "documents_unfetched": len(unfetched),
        "documents_matching": len(all_matches),
        "unfetched_matching": unfetched_matching,
        "unfetched_skipped": len(unfetched) - unfetched_matching,
        "share_kept": (unfetched_matching / len(unfetched)) if unfetched else None,
    }
