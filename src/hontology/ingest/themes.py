"""GKG themes as a code system.

The Global Knowledge Graph tags each article with themes from a published
vocabulary of tens of thousands (``CYBER_ATTACK``, ``SHORTAGE``,
``WB_167_PORTS`` …). Loaded as a second code system beside CAMEO, the same
curated concept↔code links drive the pre-scrape filter for both feeds, and the
same Filtering page curates them.

**Proposing links.** Similarity proposals consider only themes an event class
could plausibly mean. ``TAX_`` themes are excluded: they are entity lists
(occupations, languages, species, parties), and linking one to an event class
would admit every article that mentions a president or rice. So are themes used
fewer than `MIN_PROPOSAL_USES` times, the long tail of near-duplicates. That
leaves roughly two thousand of the tens of thousands, embedded as readable words
("ports", not ``WB_167_PORTS (24,531,937 uses)``), since an embedding model
compares meanings, and a code with a usage count attached has little. Any theme
can still be linked by hand.

Themes are flat — there is no hierarchy to wire — and the lookup's second column
is a usage count, kept as the code's name so curation can see how common a
theme is before linking it.
"""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from hontology.db.models import Code
from hontology.ingest import cameo

THEMES_SLUG = "gkg-themes"
THEMES_LEVEL = "theme"
THEMES_LOOKUP_URL = "https://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT"
MIN_PROPOSAL_USES = 10_000

_USES = re.compile(r"\(([\d,]+) uses\)\s*$")
# Prefixes that carry no meaning of their own: a World Bank topic number, a
# CrisisLex category id, a source tag.
_OPAQUE_PREFIX = re.compile(
    r"^(?:WB_\d+_|CRISISLEX_[A-Z]\d+_|CRISISLEX_|UNGP_|EPU_|SLFID_|USPEC_)"
)


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


def uses(name: str | None) -> int | None:
    """The usage count kept in a theme's name, if there is one."""
    match = _USES.search(name or "")
    return int(match.group(1).replace(",", "")) if match else None


def readable(theme: str) -> str:
    """``WB_167_PORTS`` → ``ports``; ``MANMADE_DISASTER_POWER_OUTAGE`` →
    ``manmade disaster power outage``."""
    return _OPAQUE_PREFIX.sub("", theme).replace("_", " ").lower().strip()


def proposal_text(theme: str, name: str | None) -> str | None:
    """What a similarity run embeds for a theme, or None if it is not proposable."""
    if theme.startswith("TAX_") or (uses(name) or 0) < MIN_PROPOSAL_USES:
        return None
    return readable(theme) or None


def fetch_lookup(url: str = THEMES_LOOKUP_URL, *, timeout: float = 60.0) -> str:
    return cameo.fetch_lookup(url, timeout=timeout)


def load_themes(session: Session, rows: list[tuple[str, int]], *, source_url: str) -> dict:
    """Upsert the theme system and its codes. Idempotent, like the CAMEO loader."""
    system = cameo.get_or_create_system(session, THEMES_SLUG, "GDELT GKG themes", source_url)
    existing = cameo.codes_by_string(session, system.id)
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
