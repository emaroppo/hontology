"""A judge is scored on what it was shown, not on what retrieval lost.

Every prompt mode is scored the same way: on the labelled pairs it showed a
deciding call. A leaf a top-down judge never reached was not shown, so it is
retrieval's miss; a shown pair whose call failed is the judge's "no".
"""

from __future__ import annotations

import pytest

from hontology.db.models import ArticleRetrieval, Document, Run, ShownLeaf, Verdict
from hontology.db.session import session_scope
from hontology.evaluation.stages import judge_eval
from hontology.ontology import portable, service

pytestmark = pytest.mark.requires_db


def _run(session, ontology_id: int, prompt_id: str) -> Run:
    run = Run(
        name=prompt_id,
        ontology_id=ontology_id,
        ontology_version="v1",
        config={"judge": {"prompt_id": prompt_id}},
        candidates_key="c",
        judge_key=f"j_{prompt_id}",
        status="done",
    )
    session.add(run)
    session.flush()
    return run


def _shown(session, run: Run, document_id: int, concept_ids: list[int]) -> None:
    session.add(ArticleRetrieval(run_id=run.id, document_id=document_id, strategy="test"))
    session.add_all(
        ShownLeaf(run_id=run.id, document_id=document_id, concept_id=c, offer=0)
        for c in concept_ids
    )


@pytest.fixture
def world():
    """Leaves Strike and Flood under Disruption; one article where both are true.

    Both judges were shown Strike only (the flat one by its cutoff, the top-down
    one by routing) and said yes to it.
    """
    with session_scope() as session:
        ontology = portable.import_ontology(
            session,
            {
                "export_version": 2,
                "slug": "test-judge-eval",
                "name": "Judge eval",
                "concepts": [{"name": n} for n in ("Disruption", "Strike", "Flood")],
                "relations": [
                    ["Strike", "subclass_of", "Disruption"],
                    ["Flood", "subclass_of", "Disruption"],
                ],
            },
        )
        ids = {c.name: c.id for c in service.list_concepts(session, ontology.id)}
        document = Document(url="https://je.test/a", url_hash="jetest00000a")
        session.add(document)
        session.flush()
        flat = _run(session, ontology.id, "strict_v1")
        top = _run(session, ontology.id, "hier_batch_v2")
        for run in (flat, top):
            _shown(session, run, document.id, [ids["Strike"]])
            session.add(
                Verdict(
                    run_id=run.id,
                    document_id=document.id,
                    concept_id=ids["Strike"],
                    matched=True,
                )
            )
        session.flush()
        truth = {(document.id, ids["Strike"]): True, (document.id, ids["Flood"]): True}
        return {
            "flat": flat.id,
            "top": top.id,
            "truth": truth,
            "ids": ids,
            "document": document.id,
            "ontology": ontology.id,
        }


@pytest.mark.parametrize("arm", ["flat", "top"])
def test_a_judge_is_not_blamed_for_a_leaf_it_was_never_shown(world, arm):
    with session_scope() as session:
        result = judge_eval.evaluate(session, [world[arm]], world["truth"])
    pooled = result["pooled"]
    assert (pooled["pairs"], pooled["tp"], pooled["fn"]) == (1, 1, 0)


def test_a_shown_pair_whose_call_failed_is_a_no(world):
    with session_scope() as session:
        run = session.get(Run, world["top"])
        session.add(
            ShownLeaf(
                run_id=run.id,
                document_id=world["document"],
                concept_id=world["ids"]["Flood"],
                offer=1,
            )
        )
        session.add(
            Verdict(
                run_id=run.id,
                document_id=world["document"],
                concept_id=world["ids"]["Flood"],
                error="server fell over",
            )
        )
        session.flush()
        result = judge_eval.evaluate(session, [world["top"]], world["truth"])
    pooled = result["pooled"]
    assert (pooled["pairs"], pooled["tp"], pooled["fn"]) == (2, 1, 1)


def test_a_run_without_a_shown_record_contributes_nothing_and_says_so(world):
    with session_scope() as session:
        old = _run(session, world["ontology"], "strict_v1")
        session.add(
            Verdict(
                run_id=old.id,
                document_id=world["document"],
                concept_id=world["ids"]["Strike"],
                matched=True,
            )
        )
        session.flush()
        result = judge_eval.evaluate(session, [old.id], world["truth"])
    assert result["pooled"]["pairs"] == 0
    assert result["runs"][0]["shown_record"] is False


def test_pooled_runs_count_a_pair_once(world):
    with session_scope() as session:
        result = judge_eval.evaluate(session, [world["flat"], world["top"]], world["truth"])
    assert result["pooled"]["pairs"] == 1
    assert [r["run_id"] for r in result["runs"]] == [world["top"], world["flat"]]
