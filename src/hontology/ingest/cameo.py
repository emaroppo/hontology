"""Ingest the CAMEO event codebook.

CAMEO is the taxonomy GDELT tags every event with, and it is public — the lookup
is a two-column tab-separated file published by the GDELT project. It is fetched
at setup time rather than vendored, so the repo carries no copy of someone else's
reference data.

The codes are hierarchical by string length:

    2 digits  root   e.g. 14      PROTEST
    3 digits  base   e.g. 145     Protest violently, riot
    4 digits  event  e.g. 1451    Engage in political dissent, riot

The parent of a code is its own prefix, which is why `Code.parent_code_id` can be
wired up from the codes alone with no extra source.
"""

from __future__ import annotations

import csv
import io

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Code, CodeSystem

CAMEO_SLUG = "cameo"
CAMEO_LOOKUP_URL = "https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt"

# Length of the code string → level name.
LEVELS = {2: "root", 3: "base", 4: "event"}


def parse_lookup(text: str) -> list[tuple[str, str]]:
    """Parse the tab-separated lookup into ``[(code, description), ...]``.

    Codes are kept as strings: they are zero-padded identifiers, and parsing them
    as integers loses the leading zero that distinguishes root ``01`` from a
    one-digit code that does not exist.
    """
    rows: list[tuple[str, str]] = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue
        code, description = row[0].strip(), row[1].strip()
        if not code or not code.isdigit():
            # Skips the header line and any trailing blank rows.
            continue
        if len(code) not in LEVELS:
            continue
        rows.append((code, description))
    return rows


def fetch_lookup(url: str = CAMEO_LOOKUP_URL, *, timeout: float = 30.0) -> str:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text


def load_codes(
    session: Session, rows: list[tuple[str, str]], *, source_url: str | None = None
) -> dict:
    """Upsert the code system and its codes, then wire the parent hierarchy.

    Idempotent: re-running updates descriptions in place and adds anything new,
    so refreshing the codebook never duplicates or orphans a curated association.
    """
    system = session.scalar(select(CodeSystem).where(CodeSystem.slug == CAMEO_SLUG))
    if system is None:
        system = CodeSystem(
            slug=CAMEO_SLUG,
            name="CAMEO event codes",
            source_url=source_url or CAMEO_LOOKUP_URL,
        )
        session.add(system)
        session.flush()

    existing = {
        code.code: code
        for code in session.scalars(select(Code).where(Code.system_id == system.id))
    }

    inserted = 0
    updated = 0
    for code_str, description in rows:
        code = existing.get(code_str)
        if code is None:
            code = Code(
                system_id=system.id,
                code=code_str,
                name=description,
                level=LEVELS[len(code_str)],
            )
            session.add(code)
            existing[code_str] = code
            inserted += 1
        elif code.name != description:
            code.name = description
            updated += 1
    session.flush()

    # Second pass: a code's parent is its own prefix.
    linked = 0
    for code_str, code in existing.items():
        parent_str = code_str[:-1]
        parent = existing.get(parent_str) if len(parent_str) in LEVELS else None
        parent_id = parent.id if parent is not None else None
        if code.parent_code_id != parent_id:
            code.parent_code_id = parent_id
            linked += 1
    session.flush()

    return {
        "system_id": system.id,
        "inserted": inserted,
        "updated": updated,
        "relinked": linked,
        "total": len(existing),
    }


def ingest(session: Session, *, url: str = CAMEO_LOOKUP_URL) -> dict:
    """Fetch and load the codebook."""
    return load_codes(session, parse_lookup(fetch_lookup(url)), source_url=url)


def resolve_chain(code: str) -> list[str]:
    """The most-specific-first fallback chain for a feed's event code.

    ``"1451"`` → ``["1451", "145", "14"]``. The filter tries these in order and
    takes the first that maps to a concept.
    """
    return [code[:n] for n in sorted(LEVELS, reverse=True) if n <= len(code)]
