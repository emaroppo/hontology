"""Backfilling the shown record from verdicts.

Each test judges with the current code, which records what it showed, then
deletes that record and rebuilds it from the verdicts alone: the rebuilt shown
set must be what was recorded. Tests run in one transaction, so every verdict
shares a timestamp here; grouping by call is tested apart, with set times.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from hontology.db.models import ArticleRetrieval, Run, ShownLeaf, Verdict
from hontology.db.session import session_scope
from hontology.evaluation import shown as shown_module
from hontology.pipeline.runs import backfill
from tests.pipeline.judge.test_extract_judging import FakeJudge as EventJudge
from tests.pipeline.judge.test_hierarchical_judging import FakeJudge, world  # noqa: F401
from tests.pipeline.judge.test_stored_retrieval import judge

pytestmark = pytest.mark.requires_db


def rebuilt(world) -> tuple[set, set, backfill.Backfill]:  # noqa: F811
    """The run's recorded shown set, and the one rebuilt after deleting it."""
    with session_scope() as session:
        recorded = shown_module.shown_pairs(session, world["run"])
        session.execute(delete(ShownLeaf).where(ShownLeaf.run_id == world["run"]))
        session.execute(delete(ArticleRetrieval).where(ArticleRetrieval.run_id == world["run"]))
        run = session.get(Run, world["run"])
        assert [r.id for r in backfill.unrecorded_runs(session)] == [world["run"]]
        result = backfill.backfill_run(session, run)
        session.flush()
        return recorded, shown_module.shown_pairs(session, world["run"]), result


def _set_prompt(world, prompt_id: str) -> None:  # noqa: F811
    with session_scope() as session:
        session.get(Run, world["run"]).config = {"judge": {"prompt_id": prompt_id}}


@pytest.mark.parametrize("prompt_id", ["hier_batch_v2", "strict_batch_v1", "strict_v1"])
def test_a_rebuilt_record_is_what_the_run_recorded(world, monkeypatch, prompt_id):  # noqa: F811
    _set_prompt(world, prompt_id)
    judge(world, FakeJudge(world["names"], yes={"Trade", "Sanction"}), monkeypatch, prompt_id)
    recorded, again, result = rebuilt(world)
    assert recorded and again == recorded
    assert result.shown == len(recorded)


def test_an_extraction_run_rebuilds_the_leaves_its_events_were_weighed_against(
    world,  # noqa: F811
    monkeypatch,
):
    _set_prompt(world, "extract_v1")
    event = "Country A banned imports from country B as a sanction."
    judge(
        world,
        EventJudge(world["names"], {event: ({"Trade", "Sanction"}, "Ban")}),
        monkeypatch,
        "extract_v1",
    )
    recorded, again, result = rebuilt(world)
    assert again == recorded
    assert {world["names"][c] for _, c in again} == {"Tariff", "Ban", "Embargo"}
    assert result.offers == 1  # per article: which event each leaf was for is lost


def test_placeholder_rows_for_an_article_with_no_events_are_not_shown(world, monkeypatch):  # noqa: F811
    _set_prompt(world, "extract_v1")
    judge(world, EventJudge(world["names"], {}), monkeypatch, "extract_v1")
    recorded, again, result = rebuilt(world)
    assert recorded == again == set()
    # Hack is the ontology's top-level leaf; its "no" only carried the cost.
    assert result.not_shown == {"placeholder for an article with no events": 1}


def test_rows_written_without_a_call_are_not_shown(world, monkeypatch, tmp_path):  # noqa: F811
    _set_prompt(world, "hier_batch_v2")
    with session_scope() as session:
        from hontology.db.models import Document

        session.get(Document, world["document"]).body_path = "missing.txt"
    judge(world, FakeJudge(world["names"], yes=set()), monkeypatch)
    recorded, again, result = rebuilt(world)
    assert recorded == again == set()
    assert result.not_shown == {"no call was made": 1}


def test_verdicts_of_one_transaction_are_one_call(world):  # noqa: F811
    """A routing call, then two leaf calls: three transactions, two offers."""
    ids, start = world["ids"], datetime(2026, 10, 7, tzinfo=UTC)
    with session_scope() as session:
        run = session.get(Run, world["run"])
        run.config = {"judge": {"prompt_id": "hier_batch_v2"}}
        for second, names in (
            (0, ("Trade", "Sanction")),
            (1, ("Hack",)),
            (2, ("Tariff", "Ban")),
        ):
            for name in names:
                session.add(
                    Verdict(
                        run_id=run.id,
                        document_id=world["document"],
                        concept_id=ids[name],
                        matched=name == "Trade",
                        created_at=start + timedelta(seconds=second),
                    )
                )
        session.flush()
        result = backfill.backfill_run(session, run)
        session.flush()
        offers = session.execute(
            select(ShownLeaf.offer, ShownLeaf.concept_id).where(ShownLeaf.run_id == run.id)
        ).all()
        stored = session.scalar(
            select(ArticleRetrieval).where(ArticleRetrieval.run_id == run.id)
        )
        assert sorted(stored.asked[0]["concept_ids"]) == sorted([ids["Trade"], ids["Sanction"]])
    assert result.offers == 2
    assert {(o, world["names"][c]) for o, c in offers} == {
        (0, "Hack"),
        (1, "Tariff"),
        (1, "Ban"),
    }
