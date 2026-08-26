"""Breakdowns, error triage, ontology lint, sweeps and the filter report."""

from __future__ import annotations

import pytest

from hontology.db.models import Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit import breakdown, errors, labels, sweep
from hontology.ontology import lint, service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def scored_run():
    """One run with a known error pattern: all mistakes on a single concept."""
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-analysis", name="Analysis")
        good = service.create_concept(
            session, ontology.id, name="Riot", definition="A violent public disturbance."
        )
        bad = service.create_concept(
            session, ontology.id, name="Strike", definition="Workers withdraw labour."
        )
        docs = []
        for i in range(8):
            document = Document(url=f"https://an.test/{i}", url_hash=f"anhash{i:010d}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        run = Run(
            name="analysis",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={"judge": {"prompt_id": "strict_v1"}},
            candidates_key="ck",
            judge_key="jk_ck",
            status="done",
        )
        session.add(run)
        session.flush()

        ids = {
            "ontology": ontology.id,
            "good": good.id,
            "bad": bad.id,
            "docs": docs,
            "run": run.id,
        }

    # Riot: perfect. Strike: every prediction wrong.
    with session_scope() as session:
        for i, document_id in enumerate(ids["docs"][:4]):
            truth = i % 2 == 0
            labels.upsert_label(
                session, document_id=document_id, concept_id=ids["good"], matched=truth
            )
            session.add(
                Verdict(
                    run_id=ids["run"],
                    document_id=document_id,
                    concept_id=ids["good"],
                    matched=truth,
                    confidence=0.9,
                    evidence="a quote",
                )
            )
        for i, document_id in enumerate(ids["docs"][4:]):
            truth = i % 2 == 0
            labels.upsert_label(
                session, document_id=document_id, concept_id=ids["bad"], matched=truth
            )
            session.add(
                Verdict(
                    run_id=ids["run"],
                    document_id=document_id,
                    concept_id=ids["bad"],
                    matched=not truth,
                    confidence=0.95,
                    evidence="a misleading quote",
                )
            )
    return ids


class TestBreakdown:
    def test_a_pooled_metric_hides_what_slicing_reveals(self, scored_run):
        with session_scope() as session:
            rows = breakdown.breakdown(session, scored_run["run"], dimension="concept")

        by_name = {r["label"]: r for r in rows}
        assert by_name["Riot"]["f1"] == pytest.approx(1.0)
        # Every Strike prediction was wrong.
        assert by_name["Strike"]["tp"] == 0
        assert by_name["Strike"]["fp"] == 2
        assert by_name["Strike"]["fn"] == 2

    def test_worst_slice_comes_first(self, scored_run):
        with session_scope() as session:
            rows = breakdown.breakdown(session, scored_run["run"], dimension="concept")
        assert rows[0]["label"] == "Strike"

    def test_each_row_carries_its_own_interval_and_count(self, scored_run):
        with session_scope() as session:
            rows = breakdown.breakdown(session, scored_run["run"], dimension="concept")
        for row in rows:
            assert row["n"] > 0
            assert "precision_ci" in row

    def test_an_unknown_dimension_is_rejected(self, scored_run):
        with session_scope() as session, pytest.raises(ValueError):
            breakdown.breakdown(session, scored_run["run"], dimension="vibes")


class TestErrorTriage:
    def test_errors_carry_the_models_own_evidence(self, scored_run):
        with session_scope() as session:
            rows = errors.triage(session, scored_run["run"])

        assert rows
        assert all(r.concept_name == "Strike" for r in rows)
        assert all(r.evidence for r in rows)

    def test_false_positives_and_negatives_are_separable(self, scored_run):
        with session_scope() as session:
            fps = errors.triage(session, scored_run["run"], kind="false_positive")
            fns = errors.triage(session, scored_run["run"], kind="false_negative")
        assert all(r.kind == "false_positive" for r in fps)
        assert all(r.kind == "false_negative" for r in fns)
        assert len(fps) == 2
        assert len(fns) == 2

    def test_summary_locates_where_errors_concentrate(self, scored_run):
        with session_scope() as session:
            stats = errors.summary(session, scored_run["run"])
        assert stats["total"] == 4
        assert list(stats["by_concept"])[0] == "Strike"

    def test_correct_predictions_are_not_errors(self, scored_run):
        with session_scope() as session:
            rows = errors.triage(session, scored_run["run"])
        assert not any(r.concept_name == "Riot" for r in rows)


class TestLint:
    def test_strength_drift_is_flagged(self):
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-lint", name="Lint")
            service.create_concept(
                session,
                ontology.id,
                name="Successful negotiation",
                definition="Parties make a concrete commitment.",
                exclusion_criteria="Talks that broke down.",
            )
            result = lint.lint(session, ontology.id)

        checks = [f["check"] for f in result["findings"]]
        assert "strength_drift" in checks

    def test_a_definition_that_earns_its_name_is_not_flagged(self):
        """The lint must not cry wolf, or it stops being read."""
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-lint2", name="Lint2")
            service.create_concept(
                session,
                ontology.id,
                name="Violent clash",
                definition="Physical violence occurs between two groups.",
                exclusion_criteria="Threats without contact.",
            )
            result = lint.lint(session, ontology.id)

        assert "strength_drift" not in [f["check"] for f in result["findings"]]

    def test_a_morphological_variant_counts_as_earning_it(self):
        """'shutdown' is earned by 'halts operations'."""
        assert lint._mentions("a port halts operations", "halt")
        assert lint._mentions("physical violence occurs", "violent")
        assert not lint._mentions("something happens", "violent")

    def test_a_missing_definition_is_a_warning(self):
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-lint3", name="Lint3")
            service.create_concept(session, ontology.id, name="Bare")
            result = lint.lint(session, ontology.id)

        assert "missing_definition" in [f["check"] for f in result["findings"]]

    def test_findings_are_advisory_not_fatal(self):
        """An ontology is the user's to author; a blocking lint gets ignored."""
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-lint4", name="Lint4")
            service.create_concept(session, ontology.id, name="Bare")
            result = lint.lint(session, ontology.id)
        assert all(f["severity"] in ("warning", "info") for f in result["findings"])


class TestSweep:
    def test_expand_is_a_cross_product(self):
        cells = sweep.expand(
            {"common": {"body_limit": 1000}},
            {
                "prompt": [
                    {"judge": {"prompt_id": "a"}, "_label": "a"},
                    {"judge": {"prompt_id": "b"}, "_label": "b"},
                ],
                "k": [{"candidates": {"max_k": 2}}, {"candidates": {"max_k": 3}}],
            },
        )
        assert len(cells) == 4
        # Base settings survive into every cell.
        assert all(c[1]["common"]["body_limit"] == 1000 for c in cells)
        # `_label` is naming metadata, never config.
        assert all("_label" not in c[1] for c in cells)

    def test_prompt_axis_does_not_fork_retrieval(self, scored_run):
        """The payoff of stage keys: N prompts share one retrieval artifact."""
        with session_scope() as session:
            plan = sweep.plan(
                session,
                scored_run["ontology"],
                base={"common": {"ontology_version": "v1"}},
                axes={
                    "prompt": [
                        {"judge": {"prompt_id": "strict_v1"}, "_label": "strict"},
                        {"judge": {"prompt_id": "lenient_v1"}, "_label": "lenient"},
                    ]
                },
            )
        summary = plan.as_dict()
        assert summary["total"] == 2
        assert summary["distinct_candidate_keys"] == 1

    def test_a_retrieval_axis_does_fork(self, scored_run):
        with session_scope() as session:
            plan = sweep.plan(
                session,
                scored_run["ontology"],
                base={"common": {"ontology_version": "v1"}},
                axes={"k": [{"candidates": {"max_k": 2}}, {"candidates": {"max_k": 5}}]},
            )
        assert plan.as_dict()["distinct_candidate_keys"] == 2

    def test_an_existing_run_marks_its_cell_done(self, scored_run):
        """What makes a sweep resumable."""
        with session_scope() as session:
            run = session.get(Run, scored_run["run"])
            plan = sweep.plan(
                session,
                scored_run["ontology"],
                base={"common": {"ontology_version": "v1"}},
                axes={},
            )
            # Force the single cell to match the existing run's keys.
            plan.cells[0].candidates_key = run.candidates_key
            plan.cells[0].judge_key = run.judge_key
            plan.cells[0].existing_run_id = run.id

        assert plan.todo == []
        assert plan.as_dict()["done"] == 1

    def test_deep_merge_preserves_untouched_siblings(self):
        merged = sweep.deep_merge(
            {"judge": {"model": "m", "prompt_id": "p"}}, {"judge": {"prompt_id": "q"}}
        )
        assert merged["judge"] == {"model": "m", "prompt_id": "q"}


class TestFilterReport:
    def test_reports_nothing_to_report_without_links(self, scored_run):
        from hontology.evalkit import filter_report

        with session_scope() as session:
            result = filter_report.report(session, scored_run["ontology"])
        assert result["usable"] is False
        assert "no concept" in result["reason"]


class TestGlossary:
    def test_every_documented_metric_has_text(self):
        from hontology.ui import glossary

        assert len(glossary.GLOSSARY) > 20
        assert all(v.strip() for v in glossary.GLOSSARY.values())

    def test_a_missing_entry_is_visible_rather_than_blank(self):
        from hontology.ui import glossary

        assert glossary.help_for("nope") is None
        assert "no glossary entry" in glossary.describe("nope")

    def test_coverage_explains_its_own_denominator(self):
        from hontology.ui import glossary

        assert "denominator" in glossary.GLOSSARY["coverage"]
