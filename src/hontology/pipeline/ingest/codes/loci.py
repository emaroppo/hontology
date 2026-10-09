"""Seed the loci table from ISO 3166-1, with FIPS 10-4 codes.

GDELT identifies countries with **FIPS 10-4** codes, not ISO. The two standards
overlap in the worst possible way: several codes exist in both and mean different
countries. FIPS ``BG`` is Bangladesh while ISO ``BG`` is Bulgaria; FIPS ``BR`` is
Brazil but FIPS ``RS`` is Russia while ISO ``RS`` is Serbia. Comparing a feed's
country code against an ISO code as a plain string therefore does not fail
loudly — it silently attributes events to the wrong country, and every metric
downstream inherits the error.

This table is the single authority for the mapping, and every layer resolves
through it rather than comparing code strings directly.
"""

from __future__ import annotations

import pycountry
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Locus
from hontology.pipeline.ingest.codes.fips import ISO2_TO_FIPS


def seed_loci(session: Session) -> tuple[int, int]:
    """Insert missing countries and correct any wrong FIPS codes.

    Idempotent: safe to run on every startup or migration. Returns
    ``(inserted, corrected)``.
    """
    existing = {locus.iso3: locus for locus in session.scalars(select(Locus))}
    inserted = 0
    corrected = 0

    for country in pycountry.countries:
        iso3 = getattr(country, "alpha_3", None)
        if not iso3:
            continue
        iso2 = getattr(country, "alpha_2", None)
        fips = ISO2_TO_FIPS.get(iso2) if iso2 else None
        name = getattr(country, "common_name", None) or country.name

        locus = existing.get(iso3)
        if locus is None:
            session.add(Locus(iso3=iso3, iso2=iso2, fips=fips, name=name))
            inserted += 1
        elif locus.fips != fips or locus.iso2 != iso2:
            locus.fips = fips
            locus.iso2 = iso2
            corrected += 1

    session.flush()
    return inserted, corrected


def by_fips(session: Session) -> dict[str, Locus]:
    """``{fips: Locus}`` — the lookup the feed ingester resolves through."""
    return {
        locus.fips: locus
        for locus in session.scalars(select(Locus).where(Locus.fips.is_not(None)))
        if locus.fips
    }


def by_iso2(session: Session) -> dict[str, Locus]:
    """``{iso2: Locus}`` — the lookup for judge output, which returns ISO alpha-2."""
    return {
        locus.iso2: locus
        for locus in session.scalars(select(Locus).where(Locus.iso2.is_not(None)))
        if locus.iso2
    }
