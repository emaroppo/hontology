"""Deciding one article's events: a choosing call per offer, and the article's
verdicts and event rows derived from what retrieval found and the judge chose.

See `pipeline.judge.extract` for the arm as a whole, and
`pipeline.retrieve.strategies.extraction` for how events are found and routed.
"""

from __future__ import annotations

from hontology.db.models import ExtractedEvent, Verdict
from hontology.pipeline.judge.context import Judging
from hontology.pipeline.judge.parse import parse_choice
from hontology.pipeline.retrieve.strategies import Retrieval
from hontology.pipeline.retrieve.strategies.base import CALL_ERRORS


def decide_events(
    judging: Judging, retrieval: Retrieval, document_id: int, top: list[int]
) -> tuple[list[Verdict], list[ExtractedEvent]]:
    """Choose at most one leaf per offered event: the verdicts and event rows."""
    template = judging.template
    assert template.build_choose is not None
    # leaf -> confidence, evidence, country
    chosen: dict[int, tuple[float, str, str]] = {}
    choices: dict[int, tuple[int | None, float | None]] = {}
    for offer in retrieval.offers:
        assert offer.event is not None
        event = retrieval.events[offer.event]
        allowed = set(offer.concept_ids)
        try:
            reply = retrieval.spend.add(
                judging.ask(
                    template.choose_system,
                    template.build_choose(event.found, judging.concepts(offer.concept_ids)),
                )
            )
            choice, confidence, quote = parse_choice(reply.text, allowed)
        except CALL_ERRORS as exc:
            event.error = str(exc)[:1000]
            continue
        choices[offer.event] = (choice, confidence)
        if choice is not None and (choice not in chosen or confidence > chosen[choice][0]):
            found = event.found
            chosen[choice] = (confidence, quote or found["evidence"], found["country"])

    event_rows = []
    for ordinal, event in enumerate(retrieval.events):
        picked, stated = choices.get(ordinal, (None, None))
        if event.error:
            outcome = "error"
            judging.stats.errors += 1
        elif ordinal not in choices:
            # Nothing but "other" took it: rejected, no choosing call.
            outcome = "rejected"
        else:
            outcome = "classified" if picked is not None else "unclassified"
        found = event.found
        event_rows.append(
            ExtractedEvent(
                run_id=judging.run_id,
                document_id=document_id,
                ordinal=ordinal,
                description=found["description"],
                evidence=found["evidence"] or None,
                status=found["status"] or None,
                country=found["country"] or None,
                concept_id=picked,
                routed_through=event.routed,
                confidence=stated,
                outcome=outcome,
                error=event.error,
            )
        )
    return _verdicts(judging, retrieval, document_id, top, chosen), event_rows


def _verdicts(
    judging: Judging,
    retrieval: Retrieval,
    document_id: int,
    top: list[int],
    chosen: dict[int, tuple[float, str, str]],
) -> list[Verdict]:
    """The article's verdicts: parents from routing, offered leaves from choices."""
    rows = [
        judging.verdict(document_id, c, matched=yes)
        for c, yes in sorted(retrieval.parent_yes.items())
    ]
    for leaf in sorted(retrieval.shown):
        confidence, evidence, country = chosen.get(leaf, (None, "", ""))
        rows.append(
            judging.verdict(
                document_id,
                leaf,
                matched=leaf in chosen,
                confidence=confidence,
                evidence=evidence or None,
                locus_id=judging.iso2_to_locus.get(country) if country else None,
            )
        )
    if not rows:
        # No events, or none reached anything: the article is a no at the top,
        # and these rows carry what the extraction cost.
        rows = [
            judging.verdict(
                document_id,
                c,
                matched=False,
                evidence="no events extracted" if not retrieval.events else None,
            )
            for c in top
        ]
    return rows
