"""A judge is scored on what it was given, not on what retrieval lost.

A per-pair judge answers only selected pairs; a top-down judge answers for every
leaf of the articles it processed, and an unreached leaf is its "no". The
retrieved scope puts both on the pairs retrieval selected.
"""

from __future__ import annotations

import pytest

from hontology.db.models import Candidate, Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evaluation.stages import judge_eval
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


def _run(session, ontology_id: int, prompt_id: str, **manifest) -> Run:
    run = Run(
        name=prompt_id,
        ontology_id=ontology_id,
        ontology_version="v1",
        config={"judge": {"prompt_id": prompt_id}},
        manifest=manifest or None,
        candidates_key="c",
        judge_key=f"j_{prompt_id}",
        status="done",
    )
    session.add(run)
    session.flush()
    return run


@pytest.fixture
def world():
    """Leaves Strike and Flood under Disruption; one article where both are true.

    Retrieval selected only Strike. The per-pair judge said yes to Strike. The
    top-down judge reached Strike only and said yes: it never asked about Flood.
    """
    with session_scope() as session:
        ontology = service.import_ontology(
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
        top = _run(session, ontology.id, "hier_batch_v2", sample={"candidates_from": flat.id})
        for concept, selected in (("Strike", True), ("Flood", False)):
            session.add(
                Candidate(
                    run_id=flat.id,
                    document_id=document.id,
                    concept_id=ids[concept],
                    source="semantic",
                    score=0.5,
                    rank=1,
                    selected=selected,
                )
            )
        for run in (flat, top):
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
        return {"flat": flat.id, "top": top.id, "truth": truth}


def test_a_per_pair_judge_is_not_blamed_for_what_retrieval_dropped(world):
    with session_scope() as session:
        result = judge_eval.evaluate(session, [world["flat"]], world["truth"])
    pooled = result["pooled"]
    assert (pooled["pairs"], pooled["tp"], pooled["fn"]) == (1, 1, 0)


def test_a_top_down_judge_answers_for_every_leaf_it_could_reach(world):
    with session_scope() as session:
        result = judge_eval.evaluate(session, [world["top"]], world["truth"])
    pooled = result["pooled"]
    assert (pooled["pairs"], pooled["tp"], pooled["fn"]) == (2, 1, 1)


def test_the_retrieved_scope_compares_both_on_the_same_pairs(world):
    with session_scope() as session:
        flat = judge_eval.evaluate(session, [world["flat"]], world["truth"], scope="retrieved")
        top = judge_eval.evaluate(session, [world["top"]], world["truth"], scope="retrieved")
    assert flat["pooled"]["pairs"] == top["pooled"]["pairs"] == 1
    assert flat["pooled"]["tp"] == top["pooled"]["tp"] == 1


def test_pooled_runs_count_a_pair_once(world):
    with session_scope() as session:
        result = judge_eval.evaluate(session, [world["flat"], world["top"]], world["truth"])
    assert result["pooled"]["pairs"] == 2
    assert [r["run_id"] for r in result["runs"]] == [world["top"], world["flat"]]
