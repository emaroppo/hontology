"""CSV plumbing shared by the label bank's import and export: cells, booleans,
dates, writing rows, and the report an import returns."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date as _date

TRUTHY = {"1", "true", "t", "yes", "y", "match", "matched"}
FALSY = {"0", "false", "f", "no", "n", "nomatch", "not matched"}


@dataclass
class ImportReport:
    created: int = 0
    updated: int = 0
    skipped_existing: int = 0
    documents_created: int = 0
    unknown_concepts: list[str] = field(default_factory=list)
    bad_rows: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "created": self.created,
            "updated": self.updated,
            "skipped_existing": self.skipped_existing,
            "documents_created": self.documents_created,
            "unknown_concepts": sorted(set(self.unknown_concepts)),
            "bad_rows": self.bad_rows[:20],
            "bad_row_count": len(self.bad_rows),
        }


def parse_bool(raw: str) -> bool:
    value = (raw or "").strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    raise ValueError(f"cannot read {raw!r} as a true/false value")


def cell(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def to_csv(rows: Iterable[dict], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def parse_date(raw: str) -> _date | None:
    value = (raw or "").strip()
    return _date.fromisoformat(value) if value else None
