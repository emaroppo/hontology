"""GDELT v2 feed: slice arithmetic and file access.

GDELT v2 publishes an export file every 15 minutes, stamped on the quarter hour
in **UTC**: ``YYYYMMDDHHMMSS`` where the minute is one of 00/15/30/45 and seconds
are always ``00``. Everything about staying current reduces to arithmetic on
those stamps, so that arithmetic lives here as pure functions and is tested
without touching the network.

Three facts that shape the design:

- **Publication lags the stamp.** A slice stamped 12:15 appears some minutes
  after 12:15. Asking for the current wall-clock quarter hour reliably 404s, so
  the scheduler works from ``lastupdate.txt`` (what the feed says is ready) and
  treats a missing file as "not yet", not as an error.
- **Slices are occasionally skipped or republished.** A gap is normal. The
  ingester must be able to tell "no file was ever published" from "we never
  looked", which is why every attempt writes a status row.
- **The stamp is the identity.** Re-fetching the same stamp must be a no-op, so
  the stamp is the natural idempotency key.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import httpx

SLICE_MINUTES = 15
SLICE_INTERVAL = timedelta(minutes=SLICE_MINUTES)
KEY_FORMAT = "%Y%m%d%H%M%S"

# Columns of the v2 export table that the pipeline actually reads. The file has
# 61 unnamed columns; mirroring all of them into the database would be storing
# someone else's schema for no benefit.
COL_EVENT_ID = 0
COL_DAY = 1
COL_EVENT_CODE = 26
COL_EVENT_BASE_CODE = 27
COL_EVENT_ROOT_CODE = 28
COL_ACTION_GEO_COUNTRY = 51
COL_SOURCE_URL = 60
EXPORT_COLUMNS = 61


def slice_key(moment: datetime) -> str:
    """The stamp of the slice covering *moment*, floored to its quarter hour."""
    moment = moment.astimezone(UTC)
    floored = moment.replace(
        minute=(moment.minute // SLICE_MINUTES) * SLICE_MINUTES, second=0, microsecond=0
    )
    return floored.strftime(KEY_FORMAT)


def key_to_datetime(key: str) -> datetime:
    return datetime.strptime(key, KEY_FORMAT).replace(tzinfo=UTC)


def next_key(key: str) -> str:
    return (key_to_datetime(key) + SLICE_INTERVAL).strftime(KEY_FORMAT)


def previous_key(key: str) -> str:
    return (key_to_datetime(key) - SLICE_INTERVAL).strftime(KEY_FORMAT)


def keys_between(start: str, end: str, *, limit: int | None = None) -> list[str]:
    """Every slice stamp in ``(start, end]`` — the gap after a watermark.

    Exclusive of *start* because the watermark records the last slice already
    done. Inclusive of *end* so the newest published slice is covered.
    """
    current = key_to_datetime(start)
    stop = key_to_datetime(end)
    out: list[str] = []
    while current < stop:
        current += SLICE_INTERVAL
        out.append(current.strftime(KEY_FORMAT))
        if limit is not None and len(out) >= limit:
            break
    return out


def seconds_until_next_slice(now: datetime, *, grace_seconds: float = 90.0) -> float:
    """How long to sleep so the next wake-up lands just after a slice publishes.

    Polling on a fixed interval drifts away from the publish schedule and ends up
    asking at the least useful moment. Aligning to the quarter hour plus a grace
    period keeps every poll landing shortly after new data actually exists.
    """
    now = now.astimezone(UTC)
    boundary = (
        now.replace(
            minute=(now.minute // SLICE_MINUTES) * SLICE_MINUTES, second=0, microsecond=0
        )
        + SLICE_INTERVAL
    )
    delay = (boundary - now).total_seconds() + grace_seconds
    # If the grace period already elapsed for this boundary, aim at the next one.
    while delay <= 0:
        delay += SLICE_INTERVAL.total_seconds()
    return delay


def lag_slices(watermark: str | None, now: datetime) -> int | None:
    """How many slices behind the watermark is. ``None`` when never ingested.

    This is the operational signal that matters for a continuously running
    ingester: not "did the last run succeed" but "are we keeping up".
    """
    if watermark is None:
        return None
    return len(keys_between(watermark, slice_key(now)))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def export_url(base_url: str, key: str) -> str:
    return f"{base_url.rstrip('/')}/{key}.export.CSV.zip"


class SliceNotPublished(LookupError):
    """The feed has no file for this stamp (yet, or ever).

    Distinct from a transport failure: a skipped slice is normal operation and
    must not be retried forever or recorded as an error.
    """


def fetch_lastupdate(base_url: str, *, timeout: float = 30.0) -> str:
    """The stamp of the newest published export slice.

    ``lastupdate.txt`` holds three lines (export, mentions, GKG); each is
    ``size hash url``. The export URL's filename carries the stamp.
    """
    # The feed redirects plain HTTP to HTTPS, so redirects are followed here as
    # well as on the slice download.
    response = httpx.get(
        f"{base_url.rstrip('/')}/lastupdate.txt", timeout=timeout, follow_redirects=True
    )
    response.raise_for_status()

    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and ".export." in parts[-1]:
            return parts[-1].rsplit("/", 1)[-1].split(".")[0]
    raise ValueError("no export entry found in lastupdate.txt")


def fetch_slice(base_url: str, key: str, *, timeout: float = 60.0) -> bytes:
    """Download one export slice, returning the raw zip bytes."""
    response = httpx.get(export_url(base_url, key), timeout=timeout, follow_redirects=True)
    if response.status_code == 404:
        raise SliceNotPublished(key)
    response.raise_for_status()
    return response.content


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
