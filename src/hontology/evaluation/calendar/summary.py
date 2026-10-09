"""Aggregates over scored entries: recall, false alarms and lead times."""

from __future__ import annotations

from datetime import UTC, datetime, time

from hontology.evaluation.calendar.entry_result import EntryResult
from hontology.evaluation.calendar.events import CONTROL, POSITIVE, PRECURSOR, STAGES
from hontology.evaluation.metrics.intervals import wilson


def _rate(hits: int, n: int) -> dict:
    low, high = wilson(hits, n)
    return {"hits": hits, "n": n, "rate": (hits / n) if n else None, "ci": [low, high]}


def lead_times(results: list[EntryResult]) -> list[dict]:
    """How far ahead of its disruption each detected precursor was seen."""
    by_id = {r.entry.id: r for r in results}
    rows = []
    for result in results:
        if (
            result.entry.kind != PRECURSOR
            or result.entry.precursor_of is None
            or result.first_match is None
        ):
            continue
        target = by_id.get(result.entry.precursor_of)
        if target is None:
            # The disruption itself was not processed: no lead time to speak of.
            continue
        start = datetime.combine(target.entry.date, time(), tzinfo=UTC)
        rows.append(
            {
                "precursor": result.entry.id,
                "disruption": target.entry.id,
                # Positive = the precursor was seen before the disruption's day.
                "lead_days": round((start - result.first_match).total_seconds() / 86400, 2),
                # The same, from the earliest match a person confirmed.
                "verified_lead_days": (
                    round((start - result.first_confirmed).total_seconds() / 86400, 2)
                    if result.first_confirmed
                    else None
                ),
                "disruption_detected": target.detected,
            }
        )
    return rows


def summarize(results: list[EntryResult]) -> dict:
    """Recall, false alarms and where positives were lost, raw and verified."""

    def of(kind: str) -> list[EntryResult]:
        return [r for r in results if r.entry.kind == kind]

    def lost(kind: str) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGES}
        for r in of(kind):
            if r.lost_at:
                counts[r.lost_at] += 1
        return counts

    return {
        "event_recall": _rate(sum(r.detected for r in of(POSITIVE)), len(of(POSITIVE))),
        "precursor_recall": _rate(sum(r.detected for r in of(PRECURSOR)), len(of(PRECURSOR))),
        "false_alarm_rate": _rate(sum(r.detected for r in of(CONTROL)), len(of(CONTROL))),
        "positives_lost_at": lost(POSITIVE),
        "verified": {
            "event_recall": _rate(
                sum(r.verified is True for r in of(POSITIVE)), len(of(POSITIVE))
            ),
            "precursor_recall": _rate(
                sum(r.verified is True for r in of(PRECURSOR)), len(of(PRECURSOR))
            ),
            # A control whose match reported a real instance is a calendar
            # error: withdrawn from the denominator, and listed.
            "false_alarm_rate": _rate(
                sum(r.detected and r.verified is False for r in of(CONTROL)),
                sum(r.verified is not True for r in of(CONTROL)),
            ),
            "controls_withdrawn": [r.entry.id for r in of(CONTROL) if r.verified is True],
            "pending_review": [r.entry.id for r in results if r.verified is None],
        },
    }
