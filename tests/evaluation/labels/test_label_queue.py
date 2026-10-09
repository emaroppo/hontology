"""Labelling queue priority.

Labelling is the scarce resource, so what reaches the top of this queue decides
what the ground-truth bank ends up being able to measure.
"""

from __future__ import annotations

import random

import pytest

from hontology.db.models import Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank, confidence
from hontology.evaluation.labels.confidence import CONF_MIN_SAMPLES, usable_confidence_runs
from hontology.evaluation.labels.queue import build_queue
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def scenario():
    """Two runs judging the same pairs, so disagreement is expressible."""

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-queue", name="Queue")
        concepts = [
            service.create_concept(session, ontology.id, name=n, definition=f"{n}.").id
            for n in ("Riot", "Strike", "Flood")
        ]
        docs = []
        for i in range(30):
            document = Document(url=f"https://q.test/{i}", url_hash=f"qhash{i:011d}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        runs = []
        for name in ("run-a", "run-b"):
            run = Run(
                name=name,
                ontology_id=ontology.id,
                ontology_version="v1",
                config={},
                candidates_key="k",
                judge_key="j_k",
                status="done",
            )
            session.add(run)
            session.flush()
            runs.append(run.id)

        ids = {"ontology": ontology.id, "concepts": concepts, "docs": docs, "runs": runs}
    yield ids


def add_verdict(session, run_id, document_id, concept_id, matched, confidence):
    session.add(
        Verdict(
            run_id=run_id,
            document_id=document_id,
            concept_id=concept_id,
            matched=matched,
            confidence=confidence,
        )
    )


class TestUsableConfidence:
    def test_a_run_that_always_says_the_same_thing_is_rejected(self, scenario):
        """A run reporting 0.5 everywhere would look maximally uncertain on every
        pair and flood the queue."""
        with session_scope() as session:
            for i in range(CONF_MIN_SAMPLES + 5):
                add_verdict(
                    session,
                    scenario["runs"][0],
                    scenario["docs"][i % 30],
                    scenario["concepts"][i % 3],
                    True,
                    0.5,
                )
        with session_scope() as session:
            assert usable_confidence_runs(session, [scenario["runs"][0]]) == set()

    def test_an_effectively_binary_run_is_rejected(self, scenario):
        """Only ever ~0 or ~1 carries no uncertainty signal."""
        with session_scope() as session:
            for i in range(CONF_MIN_SAMPLES + 5):
                add_verdict(
                    session,
                    scenario["runs"][0],
                    scenario["docs"][i % 30],
                    scenario["concepts"][i % 3],
                    i % 2 == 0,
                    0.99 if i % 2 else 0.01,
                )
        with session_scope() as session:
            assert usable_confidence_runs(session, [scenario["runs"][0]]) == set()

    def test_too_few_samples_is_rejected(self, scenario):
        with session_scope() as session:
            for i in range(3):
                add_verdict(
                    session,
                    scenario["runs"][0],
                    scenario["docs"][i],
                    scenario["concepts"][0],
                    True,
                    0.3 + i * 0.2,
                )
        with session_scope() as session:
            assert usable_confidence_runs(session, [scenario["runs"][0]]) == set()

    def test_a_spread_distribution_is_accepted(self, scenario):
        rng = random.Random(0)
        with session_scope() as session:
            for i in range(CONF_MIN_SAMPLES + 10):
                add_verdict(
                    session,
                    scenario["runs"][0],
                    scenario["docs"][i % 30],
                    scenario["concepts"][i % 3],
                    True,
                    rng.uniform(0.1, 0.9),
                )
        with session_scope() as session:
            assert usable_confidence_runs(session, [scenario["runs"][0]]) == {
                scenario["runs"][0]
            }


class TestQueuePriority:
    def test_disagreement_outranks_everything(self, scenario):
        """At least one run is wrong, and it needs no calibration to know that."""
        run_a, run_b = scenario["runs"]
        agree_doc, disagree_doc = scenario["docs"][0], scenario["docs"][1]
        concept = scenario["concepts"][0]

        with session_scope() as session:
            add_verdict(session, run_a, agree_doc, concept, True, 0.5)
            add_verdict(session, run_b, agree_doc, concept, True, 0.5)
            add_verdict(session, run_a, disagree_doc, concept, True, 0.9)
            add_verdict(session, run_b, disagree_doc, concept, False, 0.9)

        with session_scope() as session:
            items = build_queue(session, scenario["ontology"], limit=10, include_unjudged=False)
            assert items[0].document_id == disagree_doc
            assert items[0].reason == "runs disagree"

    def test_already_labelled_pairs_are_excluded(self, scenario):
        run_a = scenario["runs"][0]
        document, concept = scenario["docs"][0], scenario["concepts"][0]

        with session_scope() as session:
            add_verdict(session, run_a, document, concept, True, 0.5)
        with session_scope() as session:
            assert any(
                i.document_id == document
                for i in build_queue(
                    session, scenario["ontology"], limit=10, include_unjudged=False
                )
            )
        with session_scope() as session:
            bank.upsert_label(session, document_id=document, concept_id=concept, matched=True)
        with session_scope() as session:
            assert not any(
                i.document_id == document
                for i in build_queue(
                    session, scenario["ontology"], limit=10, include_unjudged=False
                )
            )

    def test_degenerate_confidence_does_not_drive_ranking(self, scenario):
        """The whole point of the usable-confidence check.

        Every pair has confidence 0.5 from a degenerate run. If that counted as
        uncertainty, all of them would rank as maximally informative.
        """
        run_a = scenario["runs"][0]
        with session_scope() as session:
            for i in range(CONF_MIN_SAMPLES + 5):
                add_verdict(
                    session,
                    run_a,
                    scenario["docs"][i % 30],
                    scenario["concepts"][i % 3],
                    True,
                    0.5,
                )
        with session_scope() as session:
            items = build_queue(session, scenario["ontology"], limit=50, include_unjudged=False)
            assert all(i.reason != "model uncertain" for i in items)

    def test_per_concept_cap_spreads_the_queue(self, scenario):
        """Without a cap the queue concentrates on the noisiest concept, leaving
        a bank that cannot support per-concept metrics."""
        run_a, run_b = scenario["runs"]
        with session_scope() as session:
            # Concept 0 disagrees on many documents; the others on one each.
            for i in range(10):
                add_verdict(
                    session, run_a, scenario["docs"][i], scenario["concepts"][0], True, 0.9
                )
                add_verdict(
                    session, run_b, scenario["docs"][i], scenario["concepts"][0], False, 0.9
                )
            for idx, concept in enumerate(scenario["concepts"][1:], start=20):
                add_verdict(session, run_a, scenario["docs"][idx], concept, True, 0.9)
                add_verdict(session, run_b, scenario["docs"][idx], concept, False, 0.9)

        with session_scope() as session:
            capped = build_queue(
                session,
                scenario["ontology"],
                limit=6,
                per_concept_cap=2,
                include_unjudged=False,
            )
            counts: dict[int, int] = {}
            for item in capped[:4]:
                counts[item.concept_id] = counts.get(item.concept_id, 0) + 1
            assert counts.get(scenario["concepts"][0], 0) <= 2

    def test_no_runs_yields_an_empty_queue(self, scenario):
        with session_scope() as session:
            assert build_queue(session, scenario["ontology"], run_ids=[], limit=10) == []

    def test_error_verdicts_are_ignored(self, scenario):
        """A failed pair carries no signal about the pair itself."""
        with session_scope() as session:
            session.add(
                Verdict(
                    run_id=scenario["runs"][0],
                    document_id=scenario["docs"][0],
                    concept_id=scenario["concepts"][0],
                    matched=None,
                    error="boom",
                )
            )
        with session_scope() as session:
            items = build_queue(session, scenario["ontology"], limit=10, include_unjudged=False)
            assert items == []


def test_thresholds_are_documented_constants():
    """These numbers decide what counts as signal; they should not drift silently."""
    assert confidence.CONF_MIN_SAMPLES >= 10
    assert 0 < confidence.CONF_MIN_STD < 0.5
    assert confidence.CONF_INTERIOR_LO < confidence.CONF_INTERIOR_HI
