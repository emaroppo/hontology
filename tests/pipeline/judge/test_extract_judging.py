"""Extract-then-classify judging: events first, at most one leaf per event.

A fake judge plays all three parts: it lists the events it was given, routes
each through the parents named for it, and picks the leaf named for it.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest
from sqlalchemy import select

from hontology.db.models import Candidate, Document, ExtractedEvent, Run, Verdict
from hontology.db.session import session_scope
from hontology.ontology import portable, service
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge import run as judge_module
from hontology.pipeline.judge.providers.base import Completion, ProviderError
from hontology.pipeline.runs.config import normalize

pytestmark = pytest.mark.requires_db

PAYLOAD = {
    "export_version": 2,
    "slug": "test-extract-judge",
    "name": "Extract judge",
    "concepts": [
        {"name": n, "definition": f"{n}."}
        for n in ("Trade", "Sanction", "Tariff", "Ban", "Embargo", "Hack")
    ],
    "relations": [
        ["Tariff", "subclass_of", "Trade"],
        ["Ban", "subclass_of", "Trade"],
        ["Ban", "subclass_of", "Sanction"],
        ["Embargo", "subclass_of", "Sanction"],
    ],
}
TEMPLATE = prompts.get("extract_v1")


class FakeJudge:
    """*events*: description -> (parents to route through, leaf to pick or None)."""

    def __init__(self, names: dict[int, str], events: dict[str, tuple[set[str], str | None]]):
        self.names, self.ids = names, {v: k for k, v in names.items()}
        self.events = events
        self.calls: list[str] = []
        self.fail_extraction = False

    def complete(self, *, system, prompt, config, want_json, want_reasoning, model):
        ids = [int(i) for i in re.findall(r"\[concept_id (\d+)\]", prompt)]
        if system == TEMPLATE.extract_system:
            self.calls.append("extract")
            if self.fail_extraction:
                raise ProviderError("server fell over")
            events = [
                {"description": d, "evidence": d, "status": "happened"} for d in self.events
            ]
            return self._reply({"events": events})
        description = re.search(r"description: (.*)", prompt).group(1)
        route, pick = self.events[description]
        if system in (TEMPLATE.event_top_system, TEMPLATE.event_route_system):
            self.calls.append("route")
            verdicts = [
                {"concept_id": i, "matched": self.names.get(i, "Other") in route} for i in ids
            ]
            return self._reply({"verdicts": verdicts})
        assert system == TEMPLATE.choose_system
        self.calls.append("choose")
        chosen = self.ids.get(pick) if pick else None
        return self._reply({"concept_id": chosen, "confidence": 0.9, "evidence": description})

    @staticmethod
    def _reply(body: dict) -> Completion:
        return Completion(
            text=json.dumps(body), input_tokens=101, output_tokens=7, latency_s=1.0
        )


@pytest.fixture
def world(tmp_path, monkeypatch):
    from hontology.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    (tmp_path / "ex.txt").write_text("An article about trade measures.")
    with session_scope() as session:
        ontology = portable.import_ontology(session, PAYLOAD)
        names = {c.id: c.name for c in service.list_concepts(session, ontology.id)}
        ids = {name: cid for cid, name in names.items()}
        document = Document(
            url="https://ex.test/1", url_hash="extest000001", body_path="ex.txt"
        )
        run = Run(
            name="extract",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="c",
            judge_key="j_c",
            status="running",
        )
        session.add_all([document, run])
        session.flush()
        session.add(
            Candidate(
                run_id=run.id,
                document_id=document.id,
                concept_id=ids["Ban"],
                source="semantic",
                score=0.8,
                rank=1,
                selected=True,
            )
        )
        return {"run": run.id, "names": names, "ids": ids, "document": document.id}


def judge(world, provider, monkeypatch, prompt_id="extract_v1"):
    monkeypatch.setattr(judge_module, "get_provider", lambda name, **kw: provider)
    with session_scope() as session:
        return judge_module.judge_run(
            session,
            world["run"],
            config=normalize({"judge": {"prompt_id": prompt_id}}),
            judge_body_limit=2000,
        )


def verdicts(world) -> dict[str, bool | None]:
    with session_scope() as session:
        return {
            world["names"][c]: m
            for c, m in session.execute(
                select(Verdict.concept_id, Verdict.matched).where(
                    Verdict.run_id == world["run"]
                )
            )
        }


def test_one_event_routed_through_two_families_gets_one_leaf(world, monkeypatch):
    """The smearing case: an event under both Trade and Sanction is weighed
    against every leaf it reaches, and keeps only the one it is."""
    event = "Country A banned imports from country B as a sanction."
    provider = FakeJudge(world["names"], {event: ({"Trade", "Sanction"}, "Ban")})
    judge(world, provider, monkeypatch)
    found = verdicts(world)
    assert [n for n in ("Tariff", "Ban", "Embargo") if found[n]] == ["Ban"]
    assert {n for n in ("Tariff", "Embargo") if found[n] is False} == {"Tariff", "Embargo"}
    # A top-level leaf is a routing question; not routed, it is never weighed.
    assert "Hack" not in found
    assert found["Trade"] is True and found["Sanction"] is True
    # One routing call at the top; every class below is a leaf, so it is chosen.
    assert provider.calls == ["extract", "route", "choose"]


def test_two_events_keep_two_classes(world, monkeypatch):
    """A tariff imposed and an embargo announced are two events, two classes."""
    provider = FakeJudge(
        world["names"],
        {
            "A tariff was imposed.": ({"Trade"}, "Tariff"),
            "An embargo was imposed.": ({"Sanction"}, "Embargo"),
        },
    )
    judge(world, provider, monkeypatch)
    found = verdicts(world)
    assert found["Tariff"] is True and found["Embargo"] is True
    assert found["Ban"] is False


def test_an_article_with_no_events_is_a_no_at_the_top(world, monkeypatch):
    provider = FakeJudge(world["names"], {})
    judge(world, provider, monkeypatch)
    assert verdicts(world) == {"Trade": False, "Sanction": False, "Hack": False}
    assert provider.calls == ["extract"]


def test_every_call_is_paid_for_exactly(world, monkeypatch):
    event = "Country A banned imports from country B as a sanction."
    provider = FakeJudge(world["names"], {event: ({"Trade", "Sanction"}, "Ban")})
    judge(world, provider, monkeypatch)
    with session_scope() as session:
        spent = session.execute(
            select(Verdict.input_tokens, Verdict.output_tokens).where(
                Verdict.run_id == world["run"]
            )
        ).all()
    assert sum(i for i, _ in spent) == 101 * len(provider.calls)
    assert sum(o for _, o in spent) == 7 * len(provider.calls)


def test_events_are_kept_with_their_class_and_route(world, monkeypatch):
    event = "Country A banned imports from country B as a sanction."
    judge(
        world,
        FakeJudge(world["names"], {event: ({"Trade", "Sanction"}, "Ban")}),
        monkeypatch,
    )
    with session_scope() as session:
        stored = session.scalars(
            select(ExtractedEvent).where(ExtractedEvent.run_id == world["run"])
        ).all()
        assert len(stored) == 1
        assert stored[0].concept_id == world["ids"]["Ban"]
        assert {world["names"][c] for c in stored[0].routed_through} == {"Trade", "Sanction"}
        assert stored[0].status == "happened"


def test_a_choice_outside_the_offered_classes_is_no_choice(world, monkeypatch):
    """Routed only through Trade, the event may not be called an Embargo."""
    event = "Something about trade."
    judge(world, FakeJudge(world["names"], {event: ({"Trade"}, "Embargo")}), monkeypatch)
    found = verdicts(world)
    assert not any(found.get(n) for n in ("Tariff", "Ban", "Embargo", "Hack"))


def outcomes(world) -> list[str]:
    with session_scope() as session:
        return list(
            session.scalars(
                select(ExtractedEvent.outcome)
                .where(ExtractedEvent.run_id == world["run"])
                .order_by(ExtractedEvent.ordinal)
            )
        )


def test_an_event_only_other_accepts_is_rejected_without_more_calls(world, monkeypatch):
    """Off-topic events (a war, a diplomatic row) stop at the top: one routing
    call, no choosing call, recorded as rejected."""
    provider = FakeJudge(world["names"], {"A minister resigned.": ({"Other"}, None)})
    judge(world, provider, monkeypatch)
    assert provider.calls == ["extract", "route"]
    assert outcomes(world) == ["rejected"]
    assert verdicts(world) == {"Trade": False, "Sanction": False}


def test_a_top_level_leaf_routed_yes_is_a_candidate(world, monkeypatch):
    provider = FakeJudge(world["names"], {"Ransomware stopped a port.": ({"Hack"}, "Hack")})
    judge(world, provider, monkeypatch)
    assert verdicts(world)["Hack"] is True
    assert outcomes(world) == ["classified"]


def test_an_event_weighed_but_fitting_no_leaf_is_unclassified(world, monkeypatch):
    provider = FakeJudge(world["names"], {"Trade talks stalled.": ({"Trade"}, None)})
    judge(world, provider, monkeypatch)
    assert outcomes(world) == ["unclassified"]


def test_a_restart_skips_articles_already_written(world, monkeypatch):
    event = "A tariff was imposed."
    first = FakeJudge(world["names"], {event: ({"Trade"}, "Tariff")})
    judge(world, first, monkeypatch)
    again = FakeJudge(world["names"], {event: ({"Trade"}, "Tariff")})
    result = judge(world, again, monkeypatch)
    assert again.calls == []
    assert result["skipped_already_done"] == 1


def test_a_failed_extraction_is_recorded_not_raised(world, monkeypatch):
    provider = FakeJudge(world["names"], {"x": (set(), None)})
    provider.fail_extraction = True
    result = judge(world, provider, monkeypatch)
    assert result["errors"] == 3
    with session_scope() as session:
        errors = session.scalars(
            select(Verdict.error).where(Verdict.run_id == world["run"])
        ).all()
    assert all("event extraction failed" in e for e in errors)


def test_the_top_level_leans_neither_way_and_lower_levels_lean_to_yes(world, monkeypatch):
    """Only below the top does routing lean to yes; the top separates kinds of
    event from "other" without that pull."""
    systems: list[str] = []
    provider = FakeJudge(world["names"], {"A tariff was imposed.": ({"Trade"}, "Tariff")})
    original = provider.complete

    def record(**kwargs):
        systems.append(kwargs["system"])
        return original(**kwargs)

    provider.complete = record
    judge(world, provider, monkeypatch)
    routing = [
        s for s in systems if s in (TEMPLATE.event_top_system, TEMPLATE.event_route_system)
    ]
    assert routing[0] == TEMPLATE.event_top_system
    assert "when in doubt" not in TEMPLATE.event_top_system
    assert "when in doubt" in TEMPLATE.event_route_system


class FakeEmbedder:
    """Counts each class name in a text, so a text is closest to the classes it names."""

    name = "fake"
    WORDS = ("trade", "sanction", "tariff", "ban", "embargo", "hack")

    def __init__(self) -> None:
        self.models: set[str] = set()

    def embed(self, texts, *, model):
        self.models.add(model)
        return [[1.0] + [float(t.lower().count(w)) for w in self.WORDS] for t in texts]


def judge_ranked(world, provider, monkeypatch, prompt_id="extract_embed_v1"):
    from hontology.pipeline.retrieve import embed

    embedder = FakeEmbedder()
    monkeypatch.setattr(embed, "get_provider", lambda name: embedder)
    offered: list[list[str]] = []
    original = provider.complete

    def record(**kwargs):
        if kwargs["system"] == TEMPLATE.choose_system:
            ids = re.findall(r"\[concept_id (\d+)\]", kwargs["prompt"])
            offered.append(sorted(world["names"][int(i)] for i in ids))
        return original(**kwargs)

    provider.complete = record
    judge(world, provider, monkeypatch, prompt_id=prompt_id)
    return offered, embedder


def test_ranked_leaves_come_only_from_the_branches_answered_yes(world, monkeypatch):
    """Tariff names the event too, but sits under a class the top answered no."""
    event = "A tariff and embargo story: an embargo was imposed."
    provider = FakeJudge(world["names"], {event: ({"Sanction"}, "Embargo")})
    offered, embedder = judge_ranked(world, provider, monkeypatch)
    assert offered == [["Ban", "Embargo"]]
    assert provider.calls == ["extract", "route", "choose"]
    assert embedder.models == {"mxbai-embed-large"}
    assert verdicts(world)["Embargo"] is True


def test_only_the_closest_leaves_are_offered(world, monkeypatch):
    monkeypatch.setitem(
        prompts._REGISTRY,
        "test_extract_embed_k1",
        dataclasses.replace(
            prompts.get("extract_embed_v1"), prompt_id="test_extract_embed_k1", leaf_top_k=1
        ),
    )
    event = "An embargo was imposed."
    provider = FakeJudge(world["names"], {event: ({"Trade", "Sanction"}, "Embargo")})
    offered, _ = judge_ranked(world, provider, monkeypatch, "test_extract_embed_k1")
    assert offered == [["Embargo"]]
    found = verdicts(world)
    # Leaves that were not offered are not weighed, so they get no row.
    assert found["Embargo"] is True and "Tariff" not in found and "Ban" not in found


def test_a_top_level_leaf_answered_yes_is_ranked_with_the_rest(world, monkeypatch):
    event = "A hack stopped the port."
    provider = FakeJudge(world["names"], {event: ({"Hack"}, "Hack")})
    offered, _ = judge_ranked(world, provider, monkeypatch)
    assert offered == [["Hack"]]
    assert outcomes(world) == ["classified"]


def test_an_event_only_other_accepts_is_still_rejected_before_ranking(world, monkeypatch):
    provider = FakeJudge(world["names"], {"A football match.": (set(), None)})
    offered, embedder = judge_ranked(world, provider, monkeypatch)
    assert offered == [] and provider.calls == ["extract", "route"]
    assert outcomes(world) == ["rejected"]
