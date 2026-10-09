"""Reading GDELT's zipped export and GKG files into plain records.

Only the columns the pipeline uses are named; everything else in the files is
left unread.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator

# Columns of the v2 export table that the pipeline actually reads. The file has
# 61 unnamed columns; mirroring all of them into the database would be storing
# someone else's schema for no benefit.
COL_EVENT_ID = 0
COL_DAY = 1
COL_EVENT_CODE = 26
COL_EVENT_BASE_CODE = 27
COL_EVENT_ROOT_CODE = 28
# ActionGeo_CountryCode, a FIPS code. Column 51 is ActionGeo_Type (a digit 1-5),
# which once stood here and left every event without a country.
COL_ACTION_GEO_COUNTRY = 53
COL_SOURCE_URL = 60
EXPORT_COLUMNS = 61

# GKG 2.1 columns used. Themes and locations come in a v1 form and an enhanced
# v2 form carrying character offsets; v1 is read first because the offsets are
# not needed, and v2 is the fallback when v1 is empty.
GKG_COL_RECORD_ID = 0
GKG_COL_DATE = 1
GKG_COL_COLLECTION = 2
GKG_COL_DOCUMENT = 4
GKG_COL_THEMES = 7
GKG_COL_THEMES_V2 = 8
GKG_COL_LOCATIONS = 9
GKG_COL_LOCATIONS_V2 = 10
GKG_MIN_COLUMNS = 11
# Collection 1 is the open web, the only one whose document identifier is a URL.
GKG_COLLECTION_WEB = "1"


def parse_export(payload: bytes) -> Iterator[dict]:
    """Yield the columns we use from a zipped export slice.

    Rows that are malformed or carry no source URL are skipped rather than
    raising: one bad line in a 100k-row file should not cost the whole slice.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if not names:
            return
        with archive.open(names[0]) as handle:
            stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < EXPORT_COLUMNS:
                    continue
                url = row[COL_SOURCE_URL].strip()
                if not url:
                    continue
                yield {
                    "event_id": row[COL_EVENT_ID].strip(),
                    "day": row[COL_DAY].strip(),
                    # GDELT writes the base code into the event-code column when
                    # no finer code applies. Left as-is, such a row matches at the
                    # event tier and never falls back, so the duplicate is nulled
                    # here and the tiers stay honest.
                    "event_code": (
                        row[COL_EVENT_CODE].strip()
                        if row[COL_EVENT_CODE].strip() != row[COL_EVENT_BASE_CODE].strip()
                        else ""
                    ),
                    "base_code": row[COL_EVENT_BASE_CODE].strip(),
                    "root_code": row[COL_EVENT_ROOT_CODE].strip(),
                    "country": row[COL_ACTION_GEO_COUNTRY].strip(),
                    "url": url,
                }


def _gkg_countries(field: str) -> set[str]:
    """FIPS country codes of every place a GKG location field mentions.

    Each location is ``type#name#country#...``; the country code sits third in
    both the v1 and v2 forms.
    """
    countries: set[str] = set()
    for location in field.split(";"):
        parts = location.split("#")
        if len(parts) > 2 and parts[2]:
            countries.add(parts[2])
    return countries


def _gkg_themes(v1: str, v2: str) -> set[str]:
    if v1:
        return {theme for theme in v1.split(";") if theme}
    # v2 entries are "THEME,offset".
    return {entry.split(",", 1)[0] for entry in v2.split(";") if entry}


def parse_gkg(payload: bytes) -> Iterator[dict]:
    """Yield one dict per web article in a zipped GKG slice.

    Lines are split on tabs by hand rather than with a CSV reader: GKG fields
    carry quotations with unbalanced quote marks, and a quoting-aware reader
    would silently merge the following lines into one record.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if not names:
            return
        with archive.open(names[0]) as handle:
            for raw in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
                row = raw.rstrip("\r\n").split("\t")
                if len(row) < GKG_MIN_COLUMNS or row[GKG_COL_COLLECTION] != GKG_COLLECTION_WEB:
                    continue
                url = row[GKG_COL_DOCUMENT].strip()
                if not url:
                    continue
                yield {
                    "record_id": row[GKG_COL_RECORD_ID].strip(),
                    "date": row[GKG_COL_DATE].strip(),
                    "url": url,
                    "themes": _gkg_themes(row[GKG_COL_THEMES], row[GKG_COL_THEMES_V2]),
                    "countries": _gkg_countries(row[GKG_COL_LOCATIONS])
                    or _gkg_countries(row[GKG_COL_LOCATIONS_V2]),
                }
