"""Hierarchical judging: top-down descent through the class hierarchy.

A fake judge answers yes to a chosen set of classes, so each test can say
exactly which sibling sets should be asked, and which never should.
"""

from __future__ import annotations

import json
import re

import pytest
from sqlalchemy import select

from hontology.db.models import Candidate, Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit.config import normalize
from hontology.judge import run as judge_module
from hontology.judge.providers.base import Completion, ProviderError
from hontology.ontology import service

pytestmark = pytest.mark.requires_db

# Two internal classes share the leaf "Ban" (union semantics); "Hack" is a leaf
# at the top level, with no family of its own.
PAYLOAD = {
    "export_version": 2,
    "slug": "test-hier-judge",
    "name": "Hier judge",
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


class FakeJudge:
    """Answers each listed class: yes if its name is in *yes*.

    Records the class names of every call, so the sibling sets can be checked.
    """

    def __init__(self, names: dict[int, str], yes: set[str], fail_on: set[str] | None = None):
        self.names = names
        self.yes = yes
        self.fail_on = fail_on or set()
        self.calls: list[set[str]] = []

    def complete(self, *, system, prompt, config, want_json, want_reasoning, model):
        ids = [int(i) for i in re.findall(r"\[concept_id (\d+)\]", prompt)]
        asked = {self.names[i] for i in ids}
        self.calls.append(asked)
        if asked & self.fail_on:
            raise ProviderError("server fell over")
        verdicts = [
            {"concept_id": i, "matched": self.names[i] in self.yes, "confidence": 0.9}
            for i in ids
        ]
        return Completion(
            text=json.dumps({"verdicts": verdicts}),
            input_tokens=100,
            output_tokens=10,
            latency_s=1.0,
        )


@pytest.fixture
def world(tmp_path, monkeypatch):
    """One document, gated in by a selected leaf candidate."""
    from hontology.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    (tmp_path / "hj.txt").write_text("A government banned imports as part of sanctions.")
    with session_scope() as session:
        ontology = service.import_ontology(session, PAYLOAD)
        names = {c.id: c.name for c in service.list_concepts(session, ontology.id)}
        ids = {name: cid for cid, name in names.items()}
        document = Document(
            url="https://hj.test/1", url_hash="hjtest000001", body_path="hj.txt"
        )
        run = Run(
            name="hier",
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


def judge(world, provider, monkeypatch, **kwargs):
    monkeypatch.setattr(judge_module, "get_provider", lambda name, **kw: provider)
    with session_scope() as session:
        return judge_module.judge_run(
            session,
            world["run"],
            config=normalize({"judge": {"prompt_id": "hier_batch_v1"}}),
            judge_body_limit=2000,
            **kwargs,
        )


def verdicts(world) -> dict[str, bool | None]:
    with session_scope() as session:
        return {
            world["names"][concept_id]: matched
            for concept_id, matched in session.execute(
                select(Verdict.concept_id, Verdict.matched).where(
                    Verdict.run_id == world["run"]
                )
            )
        }


def test_descent_asks_each_sibling_set_once(world, monkeypatch):
    provider = FakeJudge(world["names"], yes={"Trade", "Sanction", "Ban"})
    result = judge(world, provider, monkeypatch)
    assert provider.calls == [{"Trade", "Sanction", "Hack"}, {"Tariff", "Ban"}, {"Embargo"}]
    assert result["calls"] == 3
    assert verdicts(world) == {
        "Trade": True,
        "Sanction": True,
        "Hack": False,
        "Tariff": False,
        "Ban": True,
        "Embargo": False,
    }


def test_a_class_under_two_positive_parents_is_asked_once(world, monkeypatch):
    provider = FakeJudge(world["names"], yes={"Trade", "Sanction"})
    judge(world, provider, monkeypatch)
    assert sum("Ban" in call for call in provider.calls) == 1


def test_nothing_below_a_no_is_asked(world, monkeypatch):
    provider = FakeJudge(world["names"], yes={"Trade"})
    judge(world, provider, monkeypatch)
    assert "Embargo" not in verdicts(world)
    assert all("Embargo" not in call for call in provider.calls)


def test_nothing_below_a_failed_call_is_asked(world, monkeypatch):
    """The top-level call fails: its classes are recorded as errors, and no
    branch is opened on the strength of an answer that never came."""
    provider = FakeJudge(world["names"], yes={"Trade", "Sanction"}, fail_on={"Trade"})
    result = judge(world, provider, monkeypatch)
    assert len(provider.calls) == 1
    assert result["errors"] == 3
    assert set(verdicts(world)) == {"Trade", "Sanction", "Hack"}


def test_a_restart_finishes_without_re_asking(world, monkeypatch):
    """A crash after the top level; the resumed run continues below it and the
    final verdicts equal an uninterrupted run's."""

    class Crash(Exception):
        pass

    class CrashingJudge(FakeJudge):
        def complete(self, **kwargs):
            if self.calls:
                raise Crash("process killed")
            return super().complete(**kwargs)

    with pytest.raises(Crash):
        judge(
            world, CrashingJudge(world["names"], yes={"Trade", "Sanction", "Ban"}), monkeypatch
        )
    assert set(verdicts(world)) == {"Trade", "Sanction", "Hack"}

    resumed = FakeJudge(world["names"], yes={"Trade", "Sanction", "Ban"})
    judge(world, resumed, monkeypatch)
    assert {"Trade", "Sanction", "Hack"}.isdisjoint(set().union(*resumed.calls))
    assert set(verdicts(world)) == {"Trade", "Sanction", "Hack", "Tariff", "Ban", "Embargo"}


def test_a_document_retrieval_found_nothing_for_is_not_judged(world, monkeypatch):
    with session_scope() as session:
        for candidate in session.scalars(
            select(Candidate).where(Candidate.run_id == world["run"])
        ):
            candidate.selected = False
    provider = FakeJudge(world["names"], yes={"Trade"})
    result = judge(world, provider, monkeypatch)
    assert provider.calls == []
    assert result["documents"] == 0


def test_cost_is_recorded_per_answered_class(world, monkeypatch):
    judge(world, FakeJudge(world["names"], yes={"Trade"}), monkeypatch)
    with session_scope() as session:
        rows = session.execute(
            select(Verdict.input_tokens).where(Verdict.run_id == world["run"])
        ).all()
    # Two calls of 100 input tokens each, split exactly across their classes.
    assert sum(tokens for (tokens,) in rows) == 200


def test_reused_retrieval_copies_only_the_window_once(world):
    from hontology.evalkit.calendar_run import copy_window_candidates

    with session_scope() as session:
        source = session.get(Run, world["run"])
        target = Run(
            name="arm",
            ontology_id=source.ontology_id,
            ontology_version="v1",
            config={},
            candidates_key="c2",
            judge_key="j2_c2",
            status="running",
        )
        other = Document(url="https://hj.test/2", url_hash="hjtest000002", body_path="hj.txt")
        session.add_all([target, other])
        session.flush()
        session.add(
            Candidate(
                run_id=source.id,
                document_id=other.id,
                concept_id=world["ids"]["Tariff"],
                source="semantic",
                score=0.7,
                rank=1,
                selected=True,
            )
        )
        session.flush()
        window = {world["document"]}
        assert copy_window_candidates(session, source.id, target.id, window) == window
        assert copy_window_candidates(session, source.id, target.id, window) == window
        copied = session.scalars(select(Candidate).where(Candidate.run_id == target.id)).all()
        assert [(c.document_id, c.concept_id) for c in copied] == [
            (world["document"], world["ids"]["Ban"])
        ]


def test_reused_retrieval_refuses_a_reworded_leaf(world):
    from hontology.evalkit.calendar_run import check_same_leaves
    from hontology.ontology import snapshots

    with session_scope() as session:
        source = session.get(Run, world["run"])
        source.ontology_version = snapshots.resolve_current(session, source.ontology_id).version
        check_same_leaves(session, source, source.ontology_id)  # unchanged: fine
        service.update_concept(session, world["ids"]["Ban"], definition="Something else.")
        with pytest.raises(ValueError, match="leaves differ"):
            check_same_leaves(session, source, source.ontology_id)


def test_diagnostics_locate_hits_by_level_and_coverage(world, monkeypatch):
    from hontology.evalkit.arms import hierarchy_diagnostics

    judge(world, FakeJudge(world["names"], yes={"Trade", "Sanction", "Ban"}), monkeypatch)
    leaves = ("Tariff", "Ban", "Embargo", "Hack")
    truth = {(world["document"], world["ids"][n]): n == "Ban" for n in leaves}
    with session_scope() as session:
        ontology_id = session.get(Run, world["run"]).ontology_id
        result = hierarchy_diagnostics(session, world["run"], world["run"], ontology_id, truth)
    assert result["recall_by_level"][0] == {"n": 2, "hits": 2, "recall": 1.0}
    assert result["recall_by_level"][1] == {"n": 1, "hits": 1, "recall": 1.0}
    assert result["internal_answered_yes"] == 2
    assert result["internal_false_positives"] == 0
    assert result["coverage"] == {
        "true_positives": 1,
        "baseline_had_selected": 1,
        "beyond_baseline_retrieval": 0,
    }


def test_diagnostics_place_a_miss_at_its_level(world, monkeypatch):
    """Sanction wrongly answered no: the miss is at the top level, and the
    leaf below it still counts once reached through Trade."""
    judge(world, FakeJudge(world["names"], yes={"Trade", "Ban"}), monkeypatch)
    leaves = ("Tariff", "Ban", "Embargo", "Hack")
    truth = {(world["document"], world["ids"][n]): n == "Ban" for n in leaves}
    from hontology.evalkit.arms import hierarchy_diagnostics

    with session_scope() as session:
        ontology_id = session.get(Run, world["run"]).ontology_id
        result = hierarchy_diagnostics(session, world["run"], world["run"], ontology_id, truth)
    assert result["recall_by_level"][0] == {"n": 2, "hits": 1, "recall": 0.5}
    assert result["recall_by_level"][1]["hits"] == 1


def test_a_sample_is_judged_with_reused_retrieval_once(world, monkeypatch):
    """Only the sample's documents are judged, and a second call with the same
    documents asks nothing new."""
    from hontology.evalkit.calendar_run import judge_documents

    monkeypatch.setattr(
        judge_module,
        "get_provider",
        lambda name, **kw: FakeJudge(world["names"], yes={"Trade"}),
    )
    with session_scope() as session:
        source = session.get(Run, world["run"])
        arm = Run(
            name="arm",
            ontology_id=source.ontology_id,
            ontology_version="v1",
            config=normalize({"judge": {"prompt_id": "hier_batch_v1"}}),
            candidates_key="c3",
            judge_key="j3_c3",
            status="running",
        )
        other = Document(url="https://hj.test/3", url_hash="hjtest000003", body_path="hj.txt")
        session.add_all([arm, other])
        session.flush()
        session.add(
            Candidate(
                run_id=source.id,
                document_id=other.id,
                concept_id=world["ids"]["Tariff"],
                source="semantic",
                score=0.7,
                rank=1,
                selected=True,
            )
        )
        session.flush()
        first = judge_documents(
            session, arm, source_run_id=source.id, document_ids=[world["document"]]
        )
        again = judge_documents(
            session, arm, source_run_id=source.id, document_ids=[world["document"]]
        )
        judged = set(
            session.scalars(select(Verdict.document_id).where(Verdict.run_id == arm.id))
        )
    assert first["retrieved"] == 1
    assert first["judge"]["calls"] == 2  # top level, then Trade's children
    assert again["judge"]["calls"] == 0
    assert judged == {world["document"]}


class RecordingJudge(FakeJudge):
    """Also records which system prompt each call was given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.systems: list[str] = []

    def complete(self, *, system, prompt, **kwargs):
        self.systems.append(system)
        return super().complete(system=system, prompt=prompt, **kwargs)


def judge_v2(world, provider, monkeypatch):
    monkeypatch.setattr(judge_module, "get_provider", lambda name, **kw: provider)
    with session_scope() as session:
        return judge_module.judge_run(
            session,
            world["run"],
            config=normalize({"judge": {"prompt_id": "hier_batch_v2"}}),
            judge_body_limit=2000,
        )


def test_routing_questions_and_leaves_are_asked_in_separate_calls(world, monkeypatch):
    """The top level mixes two parents and a leaf: the parents go to a routing
    call, the leaf to a call with the flat arm's own system prompt."""
    from hontology.judge import prompts

    provider = RecordingJudge(world["names"], yes={"Trade", "Sanction", "Ban"})
    result = judge_v2(world, provider, monkeypatch)
    strict = prompts.get("strict_batch_v1").system
    route = prompts.get("hier_batch_v2").route_system
    calls = list(zip(provider.calls, provider.systems, strict=True))
    assert calls[0] == ({"Trade", "Sanction"}, route)
    assert calls[1] == ({"Hack"}, strict)
    assert all(system == strict for asked, system in calls[2:])  # only leaves below
    assert result["calls"] == 4  # routing, top leaf, Trade's leaves, Sanction's leaf
    assert verdicts(world)["Ban"] is True


def test_leaves_are_asked_exactly_as_the_flat_arm_asks_them(world, monkeypatch):
    from hontology.judge import prompts

    v2, flat = prompts.get("hier_batch_v2"), prompts.get("strict_batch_v1")
    assert v2.system == flat.system
    assert v2.build_batch is flat.build_batch


def test_a_routing_question_is_the_parents_text(world, monkeypatch):
    provider = FakeJudge(world["names"], yes=set())
    prompts_seen: list[str] = []

    def record(*, system, prompt, **kwargs):
        prompts_seen.append(prompt)
        return FakeJudge.complete(provider, system=system, prompt=prompt, **kwargs)

    monkeypatch.setattr(provider, "complete", record)
    judge_v2(world, provider, monkeypatch)
    assert "question: Trade." in prompts_seen[0]
    assert "=== Questions ===" in prompts_seen[0]
