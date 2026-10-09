"""Classifying one article's events: routing each down the hierarchy, then
choosing at most one leaf for it, and the article's verdicts derived from them.

See `pipeline.judge.extract` for the arm as a whole.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hontology.db.models import Concept, ExtractedEvent, Verdict
from hontology.pipeline.judge.context import Judging
from hontology.pipeline.judge.parse import parse_batch, parse_choice
from hontology.pipeline.judge.prompts import PromptTemplate
from hontology.pipeline.judge.providers.base import Completion, ProviderError
from hontology.pipeline.judge.wording import OTHER_ID
from hontology.pipeline.retrieve import embed, model_text


class LeafRanker:
    """Orders leaves by how close their embedding is to an event's."""

    def __init__(self, provider, model: str, leaves: list[Concept], k: int) -> None:
        self.provider, self.model, self.k = provider, model, k
        self.ids = [c.id for c in leaves]
        prefix = model_text.document_prefix(model)
        self.vectors = self._unit(
            [prefix + embed.concept_text(c, "name+definition") for c in leaves]
        )

    def _unit(self, texts: list[str]) -> np.ndarray:
        texts = [model_text.normalize_for_embedding(t) for t in texts]
        vectors = np.array(self.provider.embed(texts, model=self.model), dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def __call__(self, event: dict, allowed: set[int]) -> list[int]:
        text = f"{event['description']}\n{event['evidence']}".strip()
        scores = self.vectors @ self._unit([model_text.query_prefix(self.model) + text])[0]
        ranked = sorted(
            (i for i in range(len(self.ids)) if self.ids[i] in allowed),
            key=lambda i: (-scores[i], self.ids[i]),
        )
        return [self.ids[i] for i in ranked[: self.k]]


class Spend:
    """What one article's calls cost, to be split across its verdict rows."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0
        self.calls = 0

    def add(self, completion: Completion) -> Completion:
        self.input_tokens += completion.input_tokens or 0
        self.output_tokens += completion.output_tokens or 0
        self.seconds += completion.latency_s or 0.0
        self.calls += 1
        return completion


Call = Callable[[str | None, str, Spend], Completion]


@dataclass
class _Classes:
    """How one article's events are spread over the classes, so far."""

    parent_yes: dict[int, bool] = field(default_factory=dict)
    weighed: set[int] = field(default_factory=set)
    # leaf -> confidence, evidence, country
    chosen: dict[int, tuple[float, str, str]] = field(default_factory=dict)


@dataclass
class _Article:
    """What classifying one article's events needs."""

    judging: Judging
    document_id: int
    top: list[int]
    children: dict[int, set[int]]
    concept: Callable[[int], Concept]
    call: Call
    spend: Spend
    rank: LeafRanker | None


def classify_events(
    judging: Judging,
    events: list[dict],
    *,
    document_id: int,
    top: list[int],
    children: dict[int, set[int]],
    concept,
    call,
    spend: Spend,
    rank=None,
) -> tuple[list[Verdict], list[ExtractedEvent]]:
    """Route each event down the hierarchy and choose at most one leaf for it."""
    template = judging.template
    assert template.build_event_route is not None and template.build_choose is not None
    article = _Article(judging, document_id, top, children, concept, call, spend, rank)
    classes = _Classes()
    event_rows = [_classify_event(article, classes, n, event) for n, event in enumerate(events)]
    return _verdicts(article, classes, events), event_rows


def _classify_event(
    article: _Article, classes: _Classes, ordinal: int, event: dict
) -> ExtractedEvent:
    judging = article.judging
    template = judging.template
    assert template.build_choose is not None
    routed: list[int] = []
    choice: int | None = None
    confidence: float | None = None
    error: str | None = None
    outcome = "rejected"
    try:
        reached = _route(article, classes, event, routed)
        # Nothing but "other" took it: rejected at the top, no choosing call.
        if reached:
            classes.weighed |= reached
            reply = article.call(
                template.choose_system,
                template.build_choose(event, [article.concept(c) for c in sorted(reached)]),
                article.spend,
            )
            choice, confidence, quote = parse_choice(reply.text, reached)
            outcome = "classified" if choice is not None else "unclassified"
            chosen = classes.chosen
            if choice is not None and (choice not in chosen or confidence > chosen[choice][0]):
                chosen[choice] = (confidence, quote or event["evidence"], event["country"])
    except (ProviderError, json.JSONDecodeError) as exc:
        error = str(exc)[:1000]
        outcome = "error"
        judging.stats.errors += 1
    return ExtractedEvent(
        run_id=judging.run_id,
        document_id=article.document_id,
        ordinal=ordinal,
        description=event["description"],
        evidence=event["evidence"] or None,
        status=event["status"] or None,
        country=event["country"] or None,
        concept_id=choice,
        routed_through=routed,
        confidence=confidence,
        outcome=outcome,
        error=error,
    )


def _route(article: _Article, classes: _Classes, event: dict, routed: list[int]) -> set[int]:
    """Route *event* down from the top; the leaves it reached, to choose among.

    The parents it was routed through are appended to *routed* as they are.
    """
    template: PromptTemplate = article.judging.template
    assert template.build_event_route is not None
    children = article.children
    reached: set[int] = set()
    frontier, seen, first = list(article.top), set(), True
    while frontier:
        seen |= set(frontier)
        parents = [c for c in frontier if children.get(c)]
        # At the top every class is a routing question, leaves included,
        # beside "other"; below, a routed parent's leaves are candidates.
        asked = frontier if first else parents
        if not first:
            reached |= {c for c in frontier if not children.get(c)}
        below: set[int] = set()
        if asked:
            reply = article.call(
                template.event_top_system if first else template.event_route_system,
                template.build_event_route(
                    event, [article.concept(c) for c in asked], other=first
                ),
                article.spend,
            )
            expected = [*asked, OTHER_ID] if first else asked
            answers = parse_batch(reply.text, expected)
            for c in asked:
                yes = bool(answers[c]["matched"])
                if children.get(c):
                    classes.parent_yes[c] = classes.parent_yes.get(c, False) or yes
                    if yes:
                        routed.append(c)
                        below |= set(children[c])
                elif yes:
                    reached.add(c)
        if first and article.rank is not None:
            # Below the top, the closest leaves under the classes answered
            # yes are the candidates, with no further routing.
            allowed = reached | _leaves_under(routed, children)
            reached = set(article.rank(event, allowed)) if allowed else set()
            break
        first = False
        frontier = sorted(below - seen)
    return reached


def _verdicts(article: _Article, classes: _Classes, events: list[dict]) -> list[Verdict]:
    """The article's verdicts, derived from its events."""
    judging, document_id = article.judging, article.document_id
    rows = [
        judging.verdict(document_id, c, matched=yes)
        for c, yes in sorted(classes.parent_yes.items())
    ]
    for leaf in sorted(classes.weighed):
        confidence, evidence, country = classes.chosen.get(leaf, (None, "", ""))
        rows.append(
            judging.verdict(
                document_id,
                leaf,
                matched=leaf in classes.chosen,
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
                evidence="no events extracted" if not events else None,
            )
            for c in article.top
        ]
    return rows


def _leaves_under(classes: list[int], children: dict[int, set[int]]) -> set[int]:
    out: set[int] = set()
    stack = list(classes)
    while stack:
        c = stack.pop()
        if children.get(c):
            stack.extend(children[c])
        else:
            out.add(c)
    return out
