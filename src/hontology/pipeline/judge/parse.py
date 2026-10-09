"""Reading the judge's replies, and combining repeated ones.

Every reader here is defensive: models wrap JSON in prose or fences even when
told not to, and a reply that cannot be read raises ``json.JSONDecodeError``,
which the judging loops record as an error row rather than crash on.
"""

from __future__ import annotations

import json

from hontology.pipeline.judge import prompts
from hontology.pipeline.runs.config import AGGREGATIONS

__all__ = [
    "AGGREGATIONS",
    "STATUSES",
    "aggregate",
    "parse_batch",
    "parse_choice",
    "parse_events",
    "parse_verdict",
]

STATUSES = ("happened", "threatened", "ended")


def parse_verdict(raw: str) -> dict:
    """Parse a model response into the canonical fields.

    Falls back to the outermost ``{...}`` block: models reliably wrap JSON in
    prose or fences even when told not to, and discarding an otherwise good
    answer over a stray "Here you go:" would be throwing away real work.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])

    return _coerce(payload)


def aggregate(samples: list[dict], how: str = "majority") -> dict:
    """Combine repeated samples into one verdict.

    The vote fraction replaces the model's self-reported confidence, because it
    is the more honest signal: a model asked five times and answering yes three
    times is genuinely uncertain in a way its own stated 0.9 does not capture.

    The three rules trade precision against recall explicitly:

    - ``majority`` — the plurality answer. Balanced, and the default.
    - ``unanimous`` — matched only if *every* sample says so. Fewer, surer
      matches; the choice when a false positive is expensive.
    - ``any`` — matched if *any* sample says so. Catches concepts the model
      only occasionally notices, at the cost of precision.

    The vote fraction is reported unchanged under all three, so a verdict's
    uncertainty stays visible however the rule resolved it.
    """
    if how not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {how!r}; expected one of {AGGREGATIONS}")

    if len(samples) == 1:
        return samples[0] | {"vote_fraction": 1.0}

    positives = sum(1 for s in samples if bool(s["matched"]))
    if how == "unanimous":
        winner = positives == len(samples)
    elif how == "any":
        winner = positives > 0
    else:
        winner = positives * 2 > len(samples)

    # Always the share that agreed with the *reported* verdict.
    agreeing = positives if winner else len(samples) - positives
    fraction = agreeing / len(samples)
    representative = next((s for s in samples if bool(s["matched"]) == winner), samples[0])

    return representative | {
        "matched": winner,
        "confidence": fraction,
        "vote_fraction": fraction,
    }


def parse_batch(raw: str, expected_concept_ids: list[int]) -> dict[int, dict]:
    """Parse one batched response into ``{concept_id: verdict}``.

    Every requested concept gets an entry. A model that omits one would
    otherwise silently shrink the denominator — the pair would simply vanish
    rather than count as a miss — so anything absent is filled with a
    zero-confidence non-match and the omission is visible in the counts.
    """
    payload = json.loads(raw) if raw.strip().startswith("{") else None
    if payload is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise json.JSONDecodeError("no JSON object found", raw, 0)
        payload = json.loads(raw[start : end + 1])

    entries = payload.get("verdicts")
    if not isinstance(entries, list):
        raise json.JSONDecodeError("no 'verdicts' array in response", raw, 0)

    by_concept: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("concept_id")
        if raw_id is None:
            continue
        try:
            concept_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if concept_id not in expected_concept_ids:
            # The model invented a concept; ignore rather than store a row that
            # references nothing.
            continue
        by_concept[concept_id] = _coerce(entry) | {"omitted": False}

    for concept_id in expected_concept_ids:
        by_concept.setdefault(
            concept_id,
            {
                "matched": False,
                "confidence": 0.0,
                "country": "",
                "evidence": "",
                "omitted": True,
            },
        )
    return by_concept


def parse_events(raw: str, limit: int = prompts.MAX_EVENTS) -> list[dict]:
    """The events in an extraction reply, cleaned; at most *limit* of them."""
    payload = _json(raw)
    entries = payload.get("events")
    if not isinstance(entries, list):
        raise json.JSONDecodeError("no 'events' array in response", raw, 0)
    events: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        description = str(entry.get("description") or "").strip()
        if not description:
            continue
        status = str(entry.get("status") or "").strip().lower()
        country = str(entry.get("country") or "").strip().upper()
        events.append(
            {
                "description": description,
                "evidence": str(entry.get("evidence") or "").strip(),
                "status": status if status in STATUSES else "",
                "country": country if len(country) == 2 else "",
            }
        )
    return events[:limit]


def parse_choice(raw: str, allowed: set[int]) -> tuple[int | None, float, str]:
    """The leaf a choosing reply picked, if it is one of *allowed*."""
    payload = _json(raw)
    try:
        confidence = _unit(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = str(payload.get("evidence") or "").strip()
    chosen = payload.get("concept_id")
    try:
        concept_id = int(chosen) if chosen is not None else None
    except (TypeError, ValueError):
        concept_id = None
    # A choice outside the offered classes is no choice.
    return (concept_id if concept_id in allowed else None), confidence, evidence


def _json(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("response is not a JSON object", raw, 0)
    return payload


def _unit(value) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce(payload: dict) -> dict:
    try:
        confidence = _unit(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "matched": bool(payload.get("matched", False)),
        "confidence": confidence,
        "country": str(payload.get("country") or "").strip().upper(),
        "evidence": str(payload.get("evidence") or "").strip(),
    }
