"""Country code mapping.

The collision cases below are the entire reason `loci` is a table rather than a
string comparison, so they are asserted explicitly.
"""

from __future__ import annotations

import pytest

from hontology.db.session import session_scope
from hontology.ingest import loci


def test_fips_and_iso2_collide_with_different_meanings():
    """Comparing feed codes to ISO codes as strings mislabels these silently."""
    # ISO BG is Bulgaria; FIPS BG is Bangladesh.
    assert loci.ISO2_TO_FIPS["BG"] == "BU"  # Bulgaria's FIPS is BU, not BG
    assert loci.ISO2_TO_FIPS["BD"] == "BG"  # Bangladesh's FIPS *is* BG

    # ISO RS is Serbia; FIPS RS is Russia.
    assert loci.ISO2_TO_FIPS["RS"] == "RI"  # Serbia's FIPS is RI
    assert loci.ISO2_TO_FIPS["RU"] == "RS"  # Russia's FIPS *is* RS


def test_mapping_is_injective():
    """Two countries sharing a FIPS code would make the reverse lookup ambiguous."""
    fips = list(loci.ISO2_TO_FIPS.values())
    assert len(fips) == len(set(fips))


@pytest.mark.requires_db
def test_seed_is_idempotent_and_resolves_collisions():
    with session_scope() as session:
        loci.seed_loci(session)

    with session_scope() as session:
        inserted, _ = loci.seed_loci(session)
        assert inserted == 0

        fips = loci.by_fips(session)
        iso2 = loci.by_iso2(session)

        # The lookups disagree on these codes, which is the point.
        assert fips["BG"].name.startswith("Bangladesh")
        assert iso2["BG"].name.startswith("Bulgaria")
        assert fips["RS"].name.startswith("Russia")
        assert iso2["RS"].name.startswith("Serbia")
