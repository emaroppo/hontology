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

# ISO 3166-1 alpha-2 → FIPS 10-4, from the FIPS 10-4 standard.
ISO2_TO_FIPS: dict[str, str] = {
    "AD": "AN",
    "AE": "AE",
    "AF": "AF",
    "AG": "AC",
    "AI": "AV",
    "AL": "AL",
    "AM": "AM",
    "AO": "AO",
    "AQ": "AY",
    "AR": "AR",
    "AS": "AQ",
    "AT": "AU",
    "AU": "AS",
    "AW": "AA",
    "AZ": "AJ",
    "BA": "BK",
    "BB": "BB",
    "BD": "BG",
    "BE": "BE",
    "BF": "UV",
    "BG": "BU",
    "BH": "BA",
    "BI": "BY",
    "BJ": "BN",
    "BL": "TB",
    "BM": "BD",
    "BN": "BX",
    "BO": "BL",
    "BR": "BR",
    "BS": "BF",
    "BT": "BT",
    "BV": "BV",
    "BW": "BC",
    "BY": "BO",
    "BZ": "BH",
    "CA": "CA",
    "CC": "CK",
    "CD": "CG",
    "CF": "CT",
    "CG": "CF",
    "CH": "SZ",
    "CI": "IV",
    "CK": "CW",
    "CL": "CI",
    "CM": "CM",
    "CN": "CH",
    "CO": "CO",
    "CR": "CS",
    "CU": "CU",
    "CV": "CV",
    "CW": "UC",
    "CX": "KT",
    "CY": "CY",
    "CZ": "EZ",
    "DE": "GM",
    "DJ": "DJ",
    "DK": "DA",
    "DM": "DO",
    "DO": "DR",
    "DZ": "AG",
    "EC": "EC",
    "EE": "EN",
    "EG": "EG",
    "EH": "WI",
    "ER": "ER",
    "ES": "SP",
    "ET": "ET",
    "FI": "FI",
    "FJ": "FJ",
    "FK": "FK",
    "FM": "FM",
    "FO": "FO",
    "FR": "FR",
    "GA": "GB",
    "GB": "UK",
    "GD": "GJ",
    "GE": "GG",
    "GF": "FG",
    "GG": "GK",
    "GH": "GH",
    "GI": "GI",
    "GL": "GL",
    "GM": "GA",
    "GN": "GV",
    "GP": "GP",
    "GQ": "EK",
    "GR": "GR",
    "GS": "SX",
    "GT": "GT",
    "GU": "GQ",
    "GW": "PU",
    "GY": "GY",
    "HK": "HK",
    "HM": "HM",
    "HN": "HO",
    "HR": "HR",
    "HT": "HA",
    "HU": "HU",
    "ID": "ID",
    "IE": "EI",
    "IL": "IS",
    "IM": "IM",
    "IN": "IN",
    "IO": "IO",
    "IQ": "IZ",
    "IR": "IR",
    "IS": "IC",
    "IT": "IT",
    "JE": "JE",
    "JM": "JM",
    "JO": "JO",
    "JP": "JA",
    "KE": "KE",
    "KG": "KG",
    "KH": "CB",
    "KI": "KR",
    "KM": "CN",
    "KN": "SC",
    "KP": "KN",
    "KR": "KS",
    "KW": "KU",
    "KY": "CJ",
    "KZ": "KZ",
    "LA": "LA",
    "LB": "LE",
    "LC": "ST",
    "LI": "LS",
    "LK": "CE",
    "LR": "LI",
    "LS": "LT",
    "LT": "LH",
    "LU": "LU",
    "LV": "LG",
    "LY": "LY",
    "MA": "MO",
    "MC": "MN",
    "MD": "MD",
    "ME": "MJ",
    "MF": "RN",
    "MG": "MA",
    "MH": "RM",
    "MK": "MK",
    "ML": "ML",
    "MM": "BM",
    "MN": "MG",
    "MO": "MC",
    "MP": "CQ",
    "MQ": "MB",
    "MR": "MR",
    "MS": "MH",
    "MT": "MT",
    "MU": "MP",
    "MV": "MV",
    "MW": "MI",
    "MX": "MX",
    "MY": "MY",
    "MZ": "MZ",
    "NA": "WA",
    "NC": "NC",
    "NE": "NG",
    "NF": "NF",
    "NG": "NI",
    "NI": "NU",
    "NL": "NL",
    "NO": "NO",
    "NP": "NP",
    "NR": "NR",
    "NU": "NE",
    "NZ": "NZ",
    "OM": "MU",
    "PA": "PM",
    "PE": "PE",
    "PF": "FP",
    "PG": "PP",
    "PH": "RP",
    "PK": "PK",
    "PL": "PL",
    "PM": "SB",
    "PN": "PC",
    "PR": "RQ",
    "PS": "GZ",
    "PT": "PO",
    "PW": "PS",
    "PY": "PA",
    "QA": "QA",
    "RE": "RE",
    "RO": "RO",
    "RS": "RI",
    "RU": "RS",
    "RW": "RW",
    "SA": "SA",
    "SB": "BP",
    "SC": "SE",
    "SD": "SU",
    "SE": "SW",
    "SG": "SN",
    "SH": "SH",
    "SI": "SI",
    "SJ": "SV",
    "SK": "LO",
    "SL": "SL",
    "SM": "SM",
    "SN": "SG",
    "SO": "SO",
    "SR": "NS",
    "SS": "OD",
    "ST": "TP",
    "SV": "ES",
    "SX": "NN",
    "SY": "SY",
    "SZ": "WZ",
    "TC": "TK",
    "TD": "CD",
    "TF": "FS",
    "TG": "TO",
    "TH": "TH",
    "TJ": "TI",
    "TK": "TL",
    "TL": "TT",
    "TM": "TX",
    "TN": "TS",
    "TO": "TN",
    "TR": "TU",
    "TT": "TD",
    "TV": "TV",
    "TW": "TW",
    "TZ": "TZ",
    "UA": "UP",
    "UG": "UG",
    "US": "US",
    "UY": "UY",
    "UZ": "UZ",
    "VA": "VT",
    "VC": "VC",
    "VE": "VE",
    "VG": "VI",
    "VI": "VQ",
    "VN": "VM",
    "VU": "NH",
    "WF": "WF",
    "WS": "WS",
    "YE": "YM",
    "YT": "MF",
    "ZA": "SF",
    "ZM": "ZA",
    "ZW": "ZI",
}


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
