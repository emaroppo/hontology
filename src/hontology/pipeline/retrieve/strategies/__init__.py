"""Retrieval strategies: an article in, the leaves each deciding call is shown out.

Retrieval is a sub-pipeline with fixed ends and steps that vary by strategy:

| Prompt mode | Steps |
| --- | --- |
| per-pair, per-document | embedding selection, offered as it stands |
| hierarchical | embedding gate, routing |
| extract | embedding gate, extraction, event routing, offers |
| extract, ranked | embedding gate, extraction, top-level routing, leaf ranking, offers |

The routing and extraction prompts are still part of the judge's prompt
template; the steps only ask them. See `base` for the shared types.
"""

from __future__ import annotations

from hontology.pipeline.judge import prompts
from hontology.pipeline.retrieve.strategies.base import (
    Article,
    Asked,
    Model,
    Offer,
    Retrieval,
    Step,
    Strategy,
)
from hontology.pipeline.retrieve.strategies.extraction import (
    Event,
    ExtractEvents,
    OfferEvents,
    RankLeaves,
    RouteEvents,
)
from hontology.pipeline.retrieve.strategies.ranking import LeafRanker
from hontology.pipeline.retrieve.strategies.routing import Route
from hontology.pipeline.retrieve.strategies.selection import Selected

__all__ = [
    "Article",
    "Asked",
    "Event",
    "LeafRanker",
    "Model",
    "Offer",
    "Retrieval",
    "Step",
    "Strategy",
    "Unsplittable",
    "for_template",
]


class Unsplittable(ValueError):
    """A template whose routing and deciding share one call."""


def for_template(
    model: Model,
    *,
    top: list[int],
    children: dict[int, set[int]],
    selected: Selected | None = None,
    known: dict[int, bool | None] | None = None,
    ranker: LeafRanker | None = None,
) -> Strategy:
    """The retrieval *model*'s template implies.

    *selected* is the embedding step (stored or live); a run that has already
    gated its documents may leave it out of a top-down strategy. *known* is
    parent answers already stored, for a resumed hierarchical run.
    """
    template = model.template
    steps: list[Step] = [selected] if selected is not None else []
    if template.mode in (prompts.PER_PAIR, prompts.PER_DOCUMENT):
        if selected is None or not selected.offer:
            raise ValueError("a flat strategy is its embedding selection, offered")
        return Strategy("embedding", steps)
    if template.mode == prompts.HIERARCHICAL:
        if template.build_route is None:
            raise Unsplittable(
                f"prompt {template.prompt_id!r} asks parents and leaves in one call, "
                "so its routing cannot run apart from its judging; use hier_batch_v2"
            )
        steps.append(Route(model, top, children, known or {}))
        return Strategy("routing", steps)
    assert template.mode == prompts.EXTRACT
    steps += [ExtractEvents(model, top), RouteEvents(model, top, children, ranker is not None)]
    if ranker is not None:
        steps.append(RankLeaves(ranker, children))
    steps.append(OfferEvents())
    return Strategy("extraction, ranked" if ranker else "extraction", steps)
