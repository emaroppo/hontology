"""Retrieval strategies on their own: an article in, offers out, nothing stored.

No database: classes are detached, and a fake model answers routing and
extraction calls from the class names in each prompt.
"""

from __future__ import annotations

import json
import re

import pytest

from hontology.db.models import Concept, Document
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.providers.base import Completion, ProviderError
from hontology.pipeline.retrieve import strategies
from hontology.pipeline.retrieve.strategies.selection import GATE_CLOSED, Selected

# Trade and Sanction are parents sharing Ban; Hack is a top-level leaf.
NAMES = {1: "Trade", 2: "Sanction", 3: "Tariff", 4: "Ban", 5: "Embargo", 6: "Hack"}
CHILDREN = {1: {3, 4}, 2: {4, 5}}
TOP = [1, 2, 6]
ARTICLE = strategies.Article(Document(url="https://ex.test/1", title="t"), "Some text.")


class FakeModel:
    """Says yes to the classes in *yes*; lists *events*, each routed through the
    parents (and top-level leaves) named for it."""

    def __init__(self, prompt_id: str, yes=(), events=None, fail=False):
        self.template = prompts.get(prompt_id)
        self.body_limit = 2000
        self.yes, self.events, self.fail = set(yes), events or {}, fail
        self.calls: list[set[str]] = []

    def concepts(self, ids):
        return [Concept(id=i, name=NAMES[i], definition=f"{NAMES[i]}.") for i in ids]

    def ask(self, system, prompt):
        if self.fail:
            raise ProviderError("server fell over")
        ids = [int(i) for i in re.findall(r"\[concept_id (\d+)\]", prompt)]
        if system == self.template.extract_system:
            self.calls.append({"extract"})
            found = [
                {"description": d, "evidence": d, "status": "happened"} for d in self.events
            ]
            return self._reply({"events": found})
        self.calls.append({NAMES.get(i, "Other") for i in ids})
        match = re.search(r"description: (.*)", prompt)
        yes = self.events[match.group(1)] if match else self.yes
        return self._reply(
            {"verdicts": [{"concept_id": i, "matched": NAMES.get(i) in yes} for i in ids]}
        )

    @staticmethod
    def _reply(body: dict) -> Completion:
        return Completion(
            text=json.dumps(body), input_tokens=10, output_tokens=1, latency_s=0.1
        )


def names(ids) -> list[str]:
    return sorted(NAMES[i] for i in ids)


def retrieve(model, **kwargs) -> strategies.Retrieval:
    strategy = strategies.for_template(model, top=TOP, children=CHILDREN, **kwargs)
    return strategy.retrieve(ARTICLE)


def test_routing_asks_only_parents_and_offers_each_sibling_set():
    model = FakeModel("hier_batch_v2", yes={"Trade", "Sanction"})
    retrieval = retrieve(model)
    assert model.calls == [{"Trade", "Sanction"}]
    # Ban sits under both parents and is offered once, under the first.
    assert [names(o.concept_ids) for o in retrieval.offers] == [
        ["Hack"],
        ["Ban", "Tariff"],
        ["Embargo"],
    ]
    assert names(retrieval.shown) == ["Ban", "Embargo", "Hack", "Tariff"]


def test_routing_offers_nothing_below_a_no_or_a_failure():
    assert names(retrieve(FakeModel("hier_batch_v2", yes={"Trade"})).shown) == [
        "Ban",
        "Hack",
        "Tariff",
    ]
    failed = retrieve(FakeModel("hier_batch_v2", yes={"Trade"}, fail=True))
    assert failed.asked[0].error and names(failed.shown) == ["Hack"]


def test_known_parent_answers_are_not_asked_again():
    model = FakeModel("hier_batch_v2", yes=set())
    retrieval = retrieve(model, known={1: True, 2: False})
    assert model.calls == []
    assert names(retrieval.shown) == ["Ban", "Hack", "Tariff"]


def test_a_prompt_that_routes_and_decides_together_has_no_strategy():
    with pytest.raises(strategies.Unsplittable):
        retrieve(FakeModel("hier_batch_v1"))


def test_extraction_offers_each_event_the_leaves_it_reached():
    events = {
        "A tariff and a ban.": {"Trade"},
        "A minister resigned.": {"Other"},
    }
    retrieval = retrieve(FakeModel("extract_v1", events=events))
    assert [(o.event, names(o.concept_ids)) for o in retrieval.offers] == [
        (0, ["Ban", "Tariff"])
    ]
    assert retrieval.parent_yes == {1: True, 2: False}
    assert [e.reached for e in retrieval.events][1] == set()


def test_a_failed_extraction_stops_with_nothing_offered():
    retrieval = retrieve(FakeModel("extract_v1", fail=True))
    assert retrieval.stopped and "event extraction failed" in retrieval.error
    assert retrieval.offers == []


def test_a_flat_strategy_offers_its_selection_and_a_closed_gate_stops():
    model = FakeModel("strict_batch_v1")
    offered = retrieve(model, selected=Selected(lambda a: [4, 3], offer=True))
    assert [o.concept_ids for o in offered.offers] == [[3, 4]]
    gated = retrieve(FakeModel("hier_batch_v2"), selected=Selected(lambda a: [], offer=False))
    assert gated.stopped == GATE_CLOSED and gated.offers == []


def test_no_text_stops_before_any_step():
    model = FakeModel("hier_batch_v2", yes={"Trade"})
    strategy = strategies.for_template(model, top=TOP, children=CHILDREN)
    retrieval = strategy.retrieve(strategies.Article(ARTICLE.document, None))
    assert retrieval.error and model.calls == []
