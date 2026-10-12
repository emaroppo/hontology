"""Retrieval as a stored stage: saved before deciding, adopted by key, and the
shown record every mode leaves behind.

Reuses the hierarchical tests' ontology (Trade and Sanction share Ban; Hack is
a top-level leaf), their one gated article and their fake judge.
"""

from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import select

from hontology.db.models import ArticleRetrieval, Candidate, Run, ShownLeaf, Verdict
from hontology.db.session import session_scope
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge import run as judge_module
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs.config import normalize
from tests.pipeline.judge.test_extract_judging import FakeJudge as EventJudge
from tests.pipeline.judge.test_hierarchical_judging import FakeJudge, world  # noqa: F401

pytestmark = pytest.mark.requires_db


def judge(world, provider, monkeypatch, prompt_id="hier_batch_v2", run_id=None):  # noqa: F811
    monkeypatch.setattr(judge_module, "get_provider", lambda name, **kw: provider)
    with session_scope() as session:
        return judge_module.judge_run(
            session,
            run_id or world["run"],
            config=normalize({"judge": {"prompt_id": prompt_id}}),
            judge_body_limit=2000,
        )


def shown(world, run_id=None) -> list[tuple[int, list[str]]]:  # noqa: F811
    """Each offer of the article, in order, as (ordinal, class names)."""
    with session_scope() as session:
        rows = session.execute(
            select(ShownLeaf.offer, ShownLeaf.concept_id)
            .where(ShownLeaf.run_id == (run_id or world["run"]))
            .order_by(ShownLeaf.offer, ShownLeaf.concept_id)
        ).all()
    offers: dict[int, list[str]] = {}
    for offer, concept_id in rows:
        offers.setdefault(offer, []).append(world["names"][concept_id])
    return [(n, sorted(names)) for n, names in offers.items()]


def test_a_routed_article_is_stored_with_what_each_call_was_shown(world, monkeypatch):  # noqa: F811
    judge(world, FakeJudge(world["names"], yes={"Trade", "Sanction"}), monkeypatch)
    assert shown(world) == [(0, ["Hack"]), (1, ["Ban", "Tariff"]), (2, ["Embargo"])]
    with session_scope() as session:
        stored = session.scalar(
            select(ArticleRetrieval).where(ArticleRetrieval.run_id == world["run"])
        )
        assert stored.strategy == "routing" and stored.calls == 1
        assert [a["concept_ids"] for a in stored.asked] == [
            sorted([world["ids"]["Trade"], world["ids"]["Sanction"]])
        ]


def _second_run(world, prompt_id="hier_batch_v2") -> int:  # noqa: F811
    """Another run over the same article, sharing the first's retrieval key."""
    with session_scope() as session:
        first = session.get(Run, world["run"])
        first.retrieval_key = "r-shared"
        second = Run(
            name="second",
            ontology_id=first.ontology_id,
            ontology_version="v1",
            config=normalize({"judge": {"prompt_id": prompt_id}}),
            candidates_key="c",
            retrieval_key="r-shared",
            judge_key="j2_r-shared",
            status="running",
        )
        session.add(second)
        session.flush()
        for candidate in session.scalars(select(Candidate).where(Candidate.run_id == first.id)):
            session.add(
                Candidate(
                    run_id=second.id,
                    document_id=candidate.document_id,
                    concept_id=candidate.concept_id,
                    source=candidate.source,
                    score=candidate.score,
                    rank=candidate.rank,
                    selected=candidate.selected,
                )
            )
        return second.id


def test_a_run_sharing_the_retrieval_key_decides_on_the_same_shown_leaves(world, monkeypatch):  # noqa: F811
    judge(world, FakeJudge(world["names"], yes={"Trade", "Sanction"}), monkeypatch)
    second = _second_run(world)
    # This judge would route differently; adopting the first run's retrieval,
    # it is never asked to route at all.
    provider = FakeJudge(world["names"], yes={"Trade", "Ban"})
    judge(world, provider, monkeypatch, run_id=second)
    assert {"Trade", "Sanction"} not in provider.calls
    assert provider.calls == [{"Hack"}, {"Tariff", "Ban"}, {"Embargo"}]
    assert shown(world, second) == shown(world)
    with session_scope() as session:
        adopted = session.scalar(
            select(ArticleRetrieval).where(ArticleRetrieval.run_id == second)
        )
        assert adopted.copied_from_run == world["run"]
        # The routing answers it decided under are recorded as its own rows.
        parents = dict(
            session.execute(
                select(Verdict.concept_id, Verdict.matched).where(
                    Verdict.run_id == second,
                    Verdict.concept_id.in_([world["ids"]["Trade"], world["ids"]["Sanction"]]),
                )
            ).all()
        )
    assert parents == {world["ids"]["Trade"]: True, world["ids"]["Sanction"]: True}


def test_an_extracted_article_resumes_without_extracting_again(world, monkeypatch):  # noqa: F811
    event = "A tariff was imposed."

    class Crash(Exception):
        pass

    class CrashingAtChoice(EventJudge):
        def complete(self, **kwargs):
            if kwargs["system"] == prompts.get("extract_v1").choose_system:
                raise Crash("process killed")
            return super().complete(**kwargs)

    with pytest.raises(Crash):
        judge(
            world,
            CrashingAtChoice(world["names"], {event: ({"Trade"}, "Tariff")}),
            monkeypatch,
            prompt_id="extract_v1",
        )
    resumed = EventJudge(world["names"], {event: ({"Trade"}, "Tariff")})
    judge(world, resumed, monkeypatch, prompt_id="extract_v1")
    assert resumed.calls == ["choose"]
    assert shown(world) == [(0, ["Ban", "Tariff"])]
    with session_scope() as session:
        offered = session.scalar(
            select(ShownLeaf.event).where(ShownLeaf.run_id == world["run"]).limit(1)
        )
    assert offered == 0


@pytest.mark.parametrize("prompt_id", ["strict_batch_v1", "strict_v1"])
def test_a_flat_run_records_each_deciding_call_as_an_offer(world, monkeypatch, prompt_id):  # noqa: F811
    judge(world, FakeJudge(world["names"], yes={"Ban"}), monkeypatch, prompt_id=prompt_id)
    assert shown(world) == [(0, ["Ban"])]
    with session_scope() as session:
        stored = session.scalar(
            select(ArticleRetrieval).where(ArticleRetrieval.run_id == world["run"])
        )
    assert stored.strategy == "embedding"


def _keys(prompt_id: str, **judge) -> dict:
    return run_config.stage_keys(normalize({"judge": {"prompt_id": prompt_id, **judge}}), "v1")


def test_a_flat_run_keeps_its_keys_and_a_top_down_one_keys_its_retrieval(monkeypatch):
    flat = _keys("strict_batch_v1")
    assert flat["retrieval"] == flat["candidates"]
    assert flat["judge"].endswith("_" + flat["candidates"])
    top_down = _keys("hier_batch_v2")
    assert top_down["retrieval"] != top_down["candidates"]
    assert _keys("hier_batch_v2", model="another")["retrieval"] != top_down["retrieval"]
    # A template that decides differently but routes alike shares the retrieval.
    monkeypatch.setitem(
        prompts._REGISTRY,
        "test_hier_other_leaves",
        dataclasses.replace(
            prompts.get("hier_batch_v2"), prompt_id="test_hier_other_leaves", system="Other."
        ),
    )
    other = _keys("test_hier_other_leaves")
    assert other["retrieval"] == top_down["retrieval"]
    assert other["judge"] != top_down["judge"]
