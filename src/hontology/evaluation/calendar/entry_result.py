"""One calendar entry's result: its stage counts, matches and their reviews."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from hontology.evaluation.calendar.events import STAGES, Entry


@dataclass
class EntryResult:
    entry: Entry
    stages: dict[str, int] = field(default_factory=dict)
    first_match: datetime | None = None
    # Fetched documents left after near-duplicates collapse onto one
    # representative: what retrieval and the judge actually see.
    unique: int = 0
    # Judging spent on this window's documents, across every concept.
    cost: dict = field(default_factory=dict)
    # One row per matched representative, earliest sighting first, with any
    # review of it: {url, first_seen, evidence, confirmed}.
    matches: list[dict] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.stages.get("matched", 0) > 0

    @property
    def verified(self) -> bool | None:
        """Whether a person confirmed a match: True, False, or None if pending.

        For an event or precursor, True means some matched article describes
        this very event, and False that every match was about something else.
        For a control, True means a match reported a real instance, so the
        control itself was wrong; False confirms the false alarm.
        """
        if not self.matches:
            return False
        reviews = [m["confirmed"] for m in self.matches]
        if any(r is True for r in reviews):
            return True
        if all(r is False for r in reviews):
            return False
        return None

    @property
    def first_confirmed(self) -> datetime | None:
        return min(
            (m["first_seen"] for m in self.matches if m["confirmed"] is True), default=None
        )

    @property
    def lost_at(self) -> str | None:
        """The first stage that reached zero: where this entry was lost."""
        for stage in STAGES:
            if self.stages.get(stage, 0) == 0:
                return stage
        return None

    def as_dict(self) -> dict:
        return {
            "id": self.entry.id,
            "kind": self.entry.kind,
            "concept": self.entry.concept,
            "countries": "|".join(self.entry.countries),
            "date": self.entry.date.isoformat(),
            "detected": self.detected,
            "lost_at": self.lost_at,
            "first_match": self.first_match.isoformat() if self.first_match else None,
            "verified": self.verified,
            "matches": [
                m | {"first_seen": m["first_seen"].isoformat() if m["first_seen"] else None}
                for m in self.matches
            ],
            "unique": self.unique,
            "cost": self.cost,
            **self.stages,
        }
