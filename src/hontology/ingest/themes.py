"""GKG themes as a code system.

The Global Knowledge Graph tags each article with themes from a published
vocabulary of tens of thousands (``CYBER_ATTACK``, ``SHORTAGE``,
``WB_167_PORTS`` …). Loaded as a second code system beside CAMEO, the same
curated concept↔code links drive the pre-scrape filter for both feeds, and the
same Code Links page curates them.

Themes are flat — there is no hierarchy to wire — and the lookup's second column
is a usage count, kept as the code's name so curation can see how common a
theme is before linking it.
"""

from __future__ import annotations

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Code, CodeSystem

THEMES_SLUG = "gkg-themes"
THEMES_LEVEL = "theme"
THEMES_LOOKUP_URL = "https://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT"


def parse_lookup(text: str) -> list[tuple[str, int]]:
    """``(theme, count)`` pairs from the tab-separated lookup."""
    rows: list[tuple[str, int]] = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            rows.append((parts[0], int(parts[1])))
        except ValueError:
            continue
    return rows


def fetch_lookup(url: str = THEMES_LOOKUP_URL, *, timeout: float = 60.0) -> str:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text


def load_themes(session: Session, rows: list[tuple[str, int]], *, source_url: str) -> dict:
    """Upsert the theme system and its codes. Idempotent, like the CAMEO loader."""
    system = session.scalar(select(CodeSystem).where(CodeSystem.slug == THEMES_SLUG))
    if system is None:
        system = CodeSystem(slug=THEMES_SLUG, name="GDELT GKG themes", source_url=source_url)
        session.add(system)
        session.flush()

    existing = {
        code.code: code
        for code in session.scalars(select(Code).where(Code.system_id == system.id))
    }
    inserted = 0
    for theme, count in rows:
        name = f"{theme} ({count:,} uses)"
        code = existing.get(theme)
        if code is None:
            session.add(Code(system_id=system.id, code=theme, name=name, level=THEMES_LEVEL))
            inserted += 1
        else:
            code.name = name
    session.flush()
    return {"system_id": system.id, "inserted": inserted, "total": len(existing) + inserted}


def ingest(session: Session, *, url: str = THEMES_LOOKUP_URL) -> dict:
    return load_themes(session, parse_lookup(fetch_lookup(url)), source_url=url)
