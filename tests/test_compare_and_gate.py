"""Comparison, consistency and the regression gate."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from hontology.db.models import Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit import compare, evaluate, labels, regression, warehouse
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def two_runs(tmp_path):
    def _purge():
        with session_scope() as session:
            session.execute(text("DELETE FROM ontologies WHERE slug = 'test-cmp'"))
            session.execute(text("DELETE FROM documents WHERE url LIKE 'https://cmp.test/%'"))

    _purge()
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-cmp", name="Compare")
        concept = service.create_concept(
            session, ontology.id, name="Riot", definition="A riot."
        )
        docs = []
        for i in range(6):
            document = Document(url=f"https://cmp.test/{i}", url_hash=f"cmphash{i:09d}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        runs = []
        for name in ("strict", "lenient"):
            run = Run(
                name=name,
                ontology_id=ontology.id,
                ontology_version="v1",
                config={"judge": {"provider": "ollama", "model": "m", "prompt_id": name}},
                candidates_key="ck",
                judge_key=f"jk_{name}_ck",
                status="done",
            )
            session.add(run)
            session.flush()
            runs.append(run.id)

        ids = {
            "ontology": ontology.id,
            "concept": concept.id,
            "docs": docs,
            "runs": runs,
        }
    yield ids
    _purge()


def label_all(ids, truths: list[bool]) -> None:
    with session_scope() as session:
        for document_id, matched in zip(ids["docs"], truths, strict=True):
            labels.upsert_label(
                session,
                document_id=document_id,
                concept_id=ids["concept"],
                matched=matched,
                source=labels.HUMAN,
            )


def judge_all(ids, run_index: int, predictions: list[bool]) -> None:
    with session_scope() as session:
        for document_id, predicted in zip(ids["docs"], predictions, strict=True):
            session.add(
                Verdict(
                    run_id=ids["runs"][run_index],
                    document_id=document_id,
                    concept_id=ids["concept"],
                    matched=predicted,
                    confidence=0.9,
                )
            )


class TestEvaluate:
    def test_hand_computed_confusion(self, two_runs):
        label_all(two_runs, [True, True, True, False, False, False])
        judge_all(two_runs, 0, [True, True, False, True, False, False])

        with session_scope() as session:
            result = evaluate.evaluate_run(session, two_runs["runs"][0])

        judge = result.judge
        assert (judge["tp"], judge["fn"], judge["fp"], judge["tn"]) == (2, 1, 1, 2)
        assert judge["precision"] == pytest.approx(2 / 3)
        assert judge["recall"] == pytest.approx(2 / 3)
        assert result.n_labels == 6

    def test_errored_verdicts_are_not_predictions(self, two_runs):
        label_all(two_runs, [True] * 6)
        with session_scope() as session:
            session.add(
                Verdict(
                    run_id=two_runs["runs"][0],
                    document_id=two_runs["docs"][0],
                    concept_id=two_runs["concept"],
                    matched=None,
                    error="unparseable",
                )
            )
            session.flush()
        with session_scope() as session:
            result = evaluate.evaluate_run(session, two_runs["runs"][0])
            assert result.judge["n"] == 0
            # And liveness says why, which accuracy never would.
            assert result.liveness["ok"] is False
            assert result.liveness["errors"] == 1


class TestCompare:
    def test_a_clear_winner_is_reported(self, two_runs):
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)  # all correct
        judge_all(two_runs, 1, [False] * 6)  # all wrong

        with session_scope() as session:
            result = compare.compare_runs(session, *two_runs["runs"])

        assert result.paired["only_a_correct"] == 6
        assert result.paired["only_b_correct"] == 0
        assert "strict beats lenient" in result.verdict

    def test_a_one_pair_difference_is_not_significant(self, two_runs):
        """The guard against reporting noise as an improvement."""
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        judge_all(two_runs, 1, [True, True, True, True, True, False])

        with session_scope() as session:
            result = compare.compare_runs(session, *two_runs["runs"])

        assert result.paired["discordant"] == 1
        assert "no significant difference" in result.verdict

    def test_identical_runs_report_as_identical(self, two_runs):
        label_all(two_runs, [True, False] * 3)
        judge_all(two_runs, 0, [True, False] * 3)
        judge_all(two_runs, 1, [True, False] * 3)

        with session_scope() as session:
            result = compare.compare_runs(session, *two_runs["runs"])
        assert "identical" in result.verdict

    def test_both_runs_scored_over_the_same_shared_subset(self, two_runs):
        """Otherwise the comparison is between two different test sets."""
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        # Run B judged only the first two documents.
        with session_scope() as session:
            for document_id in two_runs["docs"][:2]:
                session.add(
                    Verdict(
                        run_id=two_runs["runs"][1],
                        document_id=document_id,
                        concept_id=two_runs["concept"],
                        matched=True,
                        confidence=0.9,
                    )
                )
        with session_scope() as session:
            result = compare.compare_runs(session, *two_runs["runs"])

        assert result.n_shared_labelled == 2
        assert result.metrics_a["n"] == 2
        assert result.metrics_b["n"] == 2


class TestConsistency:
    def test_identical_reruns_are_reproducible(self, two_runs):
        judge_all(two_runs, 0, [True, False, True, False, True, False])
        judge_all(two_runs, 1, [True, False, True, False, True, False])
        with session_scope() as session:
            result = compare.determinism(session, *two_runs["runs"])
        assert result["reproducible"] is True
        assert result["agreement"] == 1.0

    def test_a_flip_makes_it_non_reproducible(self, two_runs):
        """If this fires, run-to-run noise is a floor under every A/B delta."""
        judge_all(two_runs, 0, [True] * 6)
        judge_all(two_runs, 1, [True, True, True, True, True, False])
        with session_scope() as session:
            result = compare.determinism(session, *two_runs["runs"])
        assert result["reproducible"] is False
        assert result["flipped"] == 1


class TestRegressionGate:
    def test_liveness_fails_independently_of_accuracy(self, two_runs):
        """A malformed-output bug leaves rates healthy and shrinks the denominator."""
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)

        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0])
            assert baseline.min_precision == 1.0

        # Perfect on everything it answered, but one pair errored. A pair has one
        # verdict, so an errored pair *is* that verdict — not an extra row.
        with session_scope() as session:
            verdict = session.scalar(
                select(Verdict).where(
                    Verdict.run_id == two_runs["runs"][0],
                    Verdict.document_id == two_runs["docs"][0],
                )
            )
            verdict.matched = None
            verdict.confidence = None
            verdict.error = "boom"
        with session_scope() as session:
            result = regression.check(session, two_runs["runs"][0], baseline)

        assert result["ok"] is False
        assert any("liveness" in failure for failure in result["failures"])

    def test_a_collapse_trips_the_floors(self, two_runs):
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        judge_all(two_runs, 1, [False] * 6)

        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0])
        with session_scope() as session:
            result = regression.check(session, two_runs["runs"][1], baseline)

        assert result["ok"] is False
        assert any("recall" in failure for failure in result["failures"])

    def test_tolerance_absorbs_small_drift(self, two_runs):
        """A local model is not bit-reproducible; a hard gate would be switched off."""
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        judge_all(two_runs, 1, [True, True, True, True, True, False])

        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0], tolerance=0.25)
        with session_scope() as session:
            result = regression.check(session, two_runs["runs"][1], baseline)
        assert result["ok"] is True

    def test_baseline_round_trips_through_json(self, two_runs, tmp_path):
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0])

        path = tmp_path / "baseline.json"
        regression.save_baseline(baseline, path)
        assert regression.load_baseline(path) == baseline


class TestWarehouse:
    def test_recording_is_idempotent_per_run(self, two_runs, tmp_path):
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)
        path = tmp_path / "wh.duckdb"

        with session_scope() as session:
            evaluation = evaluate.evaluate_run(session, two_runs["runs"][0])

        for _ in range(3):
            warehouse.record(
                evaluation, config={}, candidates_key="ck", judge_key="jk", path=path
            )

        rows = warehouse.leaderboard(path)
        assert len(rows) == 1
        assert rows[0]["f1"] == pytest.approx(1.0)

    def test_counts_are_stored_so_rates_can_be_recomputed(self, two_runs, tmp_path):
        """Averaging two runs' F1 is not the F1 of their union."""
        label_all(two_runs, [True, True, True, False, False, False])
        judge_all(two_runs, 0, [True, True, False, True, False, False])
        path = tmp_path / "wh.duckdb"

        with session_scope() as session:
            evaluation = evaluate.evaluate_run(session, two_runs["runs"][0])
        warehouse.record(evaluation, config={}, candidates_key="ck", judge_key="jk", path=path)

        row = warehouse.leaderboard(path)[0]
        assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (2, 1, 2, 1)


class TestEmptyBankGate:
    def test_an_empty_bank_is_inconclusive_not_a_pass(self, two_runs):
        """A gate that reports green while checking nothing is worse than no gate."""
        judge_all(two_runs, 0, [True] * 6)  # verdicts, but no labels at all

        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0])
        with session_scope() as session:
            result = regression.check(session, two_runs["runs"][0], baseline)

        assert result["inconclusive"] is True
        assert result["ok"] is False
        assert any("inconclusive" in failure for failure in result["failures"])

    def test_a_covered_run_is_conclusive(self, two_runs):
        label_all(two_runs, [True] * 6)
        judge_all(two_runs, 0, [True] * 6)

        with session_scope() as session:
            baseline = regression.capture_baseline(session, two_runs["runs"][0])
        with session_scope() as session:
            result = regression.check(session, two_runs["runs"][0], baseline)

        assert result["inconclusive"] is False
        assert result["ok"] is True
        assert result["n_judged_labelled"] == 6
