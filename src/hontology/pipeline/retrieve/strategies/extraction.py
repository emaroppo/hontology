"""Extracting an article's events and routing each: arm E's retrieval.

The article is read once, by a call that lists its distinct events. Each event
is then routed down the class hierarchy on its own description and evidence,
and the leaves it reaches become an offer: the leaves one choosing call, the
deciding call of arm E, is shown for that event.

- **Top level.** Every top-level class is a routing question, leaves included,
  beside "other", under an instruction that leans neither way. An event that
  only "other" accepts reaches nothing: no offer, and no more calls.
- **Below the top.** Routing leans to yes. A routed parent's leaves are reached
  directly; its parents are asked again, round by round.
- **Ranked instead** (`RankLeaves`). Routing stops at the top, and the event
  reaches the ``k`` leaves under the classes it answered yes to that sit
  closest to it in embedding space.

A failed extraction leaves nothing to offer and stops the retrieval; a failed
routing call costs only its own event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hontology.pipeline.judge.parse import parse_batch, parse_events
from hontology.pipeline.judge.wording import OTHER_ID
from hontology.pipeline.retrieve.strategies.base import CALL_ERRORS, Model, Offer, Retrieval
from hontology.pipeline.retrieve.strategies.ranking import LeafRanker


@dataclass
class Event:
    """One event the extraction call listed, and how retrieval placed it."""

    found: dict  # description, evidence, status, country
    # Parents it was routed through, in the order they answered yes.
    routed: list[int] = field(default_factory=list)
    reached: set[int] = field(default_factory=set)
    error: str | None = None


@dataclass
class ExtractEvents:
    model: Model
    top: list[int]
    name: str = "extraction"

    def __call__(self, retrieval: Retrieval) -> None:
        template, article = self.model.template, retrieval.article
        assert template.build_extract is not None and article.body is not None
        prompt = template.build_extract(
            article.document, self.model.concepts(self.top), article.body, self.model.body_limit
        )
        try:
            reply = retrieval.spend.add(self.model.ask(template.extract_system, prompt))
            retrieval.events = [Event(found) for found in parse_events(reply.text)]
        except CALL_ERRORS as exc:
            retrieval.stopped = retrieval.error = f"event extraction failed: {exc}"[:1000]
            return
        retrieval.note(self.name, events=len(retrieval.events))


@dataclass
class RouteEvents:
    model: Model
    top: list[int]
    children: dict[int, set[int]]
    # Stop after the top level, leaving the rest to a ranking step.
    top_only: bool = False
    name: str = "event routing"

    def __call__(self, retrieval: Retrieval) -> None:
        for event in retrieval.events:
            try:
                self._route(retrieval, event)
            except CALL_ERRORS as exc:
                event.error = str(exc)[:1000]
        retrieval.note(
            self.name,
            reached=sum(bool(e.reached) for e in retrieval.events if not e.error),
            failed=sum(e.error is not None for e in retrieval.events),
        )

    def _route(self, retrieval: Retrieval, event: Event) -> None:
        template, children = self.model.template, self.children
        assert template.build_event_route is not None
        frontier, seen, first = list(self.top), set(), True
        while frontier:
            seen |= set(frontier)
            parents = [c for c in frontier if children.get(c)]
            # At the top every class is a routing question, leaves included,
            # beside "other"; below, a routed parent's leaves are reached.
            asked = frontier if first else parents
            if not first:
                event.reached |= {c for c in frontier if not children.get(c)}
            below: set[int] = set()
            if asked:
                reply = retrieval.spend.add(
                    self.model.ask(
                        template.event_top_system if first else template.event_route_system,
                        template.build_event_route(
                            event.found, self.model.concepts(asked), other=first
                        ),
                    )
                )
                answers = parse_batch(reply.text, [*asked, OTHER_ID] if first else asked)
                for c in asked:
                    yes = bool(answers[c]["matched"])
                    if children.get(c):
                        retrieval.parent_yes[c] = retrieval.parent_yes.get(c, False) or yes
                        if yes:
                            event.routed.append(c)
                            below |= set(children[c])
                    elif yes:
                        event.reached.add(c)
            if first and self.top_only:
                return
            first = False
            frontier = sorted(below - seen)


@dataclass
class RankLeaves:
    """Replaces routing below the top: the closest leaves under the classes an
    event answered yes to."""

    ranker: LeafRanker
    children: dict[int, set[int]]
    name: str = "leaf ranking"

    def __call__(self, retrieval: Retrieval) -> None:
        for event in retrieval.events:
            if event.error:
                continue
            allowed = event.reached | leaves_under(event.routed, self.children)
            try:
                event.reached = set(self.ranker(event.found, allowed)) if allowed else set()
            except CALL_ERRORS as exc:
                event.error = str(exc)[:1000]
        retrieval.note(self.name, k=self.ranker.k)


@dataclass
class OfferEvents:
    """One offer per event that reached any leaf."""

    name: str = "offers"

    def __call__(self, retrieval: Retrieval) -> None:
        for ordinal, event in enumerate(retrieval.events):
            if event.reached and not event.error:
                retrieval.offers.append(Offer(sorted(event.reached), event=ordinal))
        retrieval.note(self.name, offers=len(retrieval.offers))


def leaves_under(classes: list[int], children: dict[int, set[int]]) -> set[int]:
    out: set[int] = set()
    stack = list(classes)
    while stack:
        c = stack.pop()
        if children.get(c):
            stack.extend(children[c])
        else:
            out.add(c)
    return out
