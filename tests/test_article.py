"""Article-level scoring, the frozen sample order, and whole-document labels."""

from __future__ import annotations

from collections import Counter

import pytest
from sqlalchemy import select

from hontology.db.models import Document, PairLabel, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit import article, sample
from hontology.evalkit.label_io import import_document_labels
from hontology.ontology import service


def docs(n: int, window: str, band: str, start: int = 0) -> list[sample.SampledDocument]:
    return [
        sample.SampledDocument(start + i, f"https://s.test/{start + i}", window, band)
        for i in range(n)
    ]


class TestFrozenOrder:
    FRAME = docs(60, "a", "high") + docs(30, "a", "none", 100) + docs(10, "b", "low", 200)

    def test_the_same_seed_gives_the_same_order(self):
        first = [d.document_id for d in sample.frozen_order(self.FRAME, seed=1)]
        again = [d.document_id for d in sample.frozen_order(self.FRAME, seed=1)]
        assert first == again
        assert first != [d.document_id for d in sample.frozen_order(self.FRAME, seed=2)]

    def test_every_prefix_keeps_the_strata_in_proportion(self):
        """Stopping after any number of documents must still give a stratified
        sample, since the stopping point is not known in advance."""
        ordered = sample.frozen_order(self.FRAME, seed=3)
        for size in (10, 20, 50):
            counts = Counter(d.stratum for d in ordered[:size])
            assert abs(counts["a/high"] - 0.6 * size) <= 1
            assert abs(counts["a/none"] - 0.3 * size) <= 1
            assert abs(counts["b/low"] - 0.1 * size) <= 1

    def test_documents_retrieval_found_nothing_in_are_a_stratum(self):
        assert sample.score_band(None) == "none"
        assert sample.score_band(0.55) == "low"
        assert sample.score_band(0.75) == "high"

    def test_the_manifest_fingerprints_its_order(self):
        ordered = sample.frozen_order(self.FRAME, seed=1)
        one = sample.manifest(1, "abc", 1, ordered)
        two = sample.manifest(1, "abc", 1, list(reversed(ordered)))
        assert one["size"] == 100
        assert one["order_sha256"] != two["order_sha256"]


class TestScoring:
    TRUTH = {(1, 10): True, (1, 11): False, (2, 10): False, (2, 11): True, (3, 10): False}

    def test_an_unjudged_pair_counts_as_no(self):
        result = article.confusion(self.TRUTH, {(1, 10): True})
        assert (result.tp, result.fn, result.fp) == (1, 1, 0)

    def test_document_bootstrap_keeps_documents_with_only_negatives(self):
        """Document 3 has no positives and no predictions; dropping it from the
        resampling pool would bias every interval."""
        scores = article.document_bootstrap(self.TRUTH, {(1, 10): True}, n_boot=200)
        assert scores["documents"] == 3
        assert scores["precision"] == 1.0
        assert scores["recall"] == 0.5

    def test_a_clearly_better_arm_is_an_improvement(self):
        truth = {(d, 1): d % 2 == 0 for d in range(200)}
        baseline = {key: False for key in truth}
        arm = dict(truth)
        result = article.paired_document_bootstrap(truth, baseline, arm, n_boot=300)
        assert (result["baseline_f1"], result["arm_f1"], result["difference"]) == (
            0.0,
            1.0,
            1.0,
        )
        assert result["improvement"] is True

    def test_identical_arms_are_not_an_improvement(self):
        truth = {(d, 1): d % 3 == 0 for d in range(150)}
        guess = {key: (key[0] % 2 == 0) for key in truth}
        result = article.paired_document_bootstrap(truth, guess, guess, n_boot=300)
        assert result["difference"] == 0
        assert result["improvement"] is False

    def test_stopping_looks_only_at_interval_width(self):
        wide = {
            "documents": 10,
            "tp": 5,
            "fn": 5,
            "precision_ci": [0.3, 0.9],
            "recall_ci": [0.2, 0.8],
        }
        narrow = {
            "documents": 900,
            "tp": 400,
            "fn": 90,
            "precision_ci": [0.70, 0.78],
            "recall_ci": [0.78, 0.86],
        }
        assert article.sample_status(wide)["target_met"] is False
        assert article.sample_status(narrow)["target_met"] is True


@pytest.mark.requires_db
class TestDocumentLabels:
    @pytest.fixture
    def setup(self):
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-doclabels", name="Docs")
            for name in ("Riot", "Strike", "Flood"):
                service.create_concept(session, ontology.id, name=name, definition=f"{name}.")
            for url in ("https://dl.test/1", "https://dl.test/2"):
                session.add(Document(url=url, url_hash=url[-12:].rjust(12, "x")))
            return ontology.id

    def _labels(self, ontology_id) -> dict[tuple[str, str], bool]:
        with session_scope() as session:
            rows = session.execute(
                select(Document.url, PairLabel.matched, PairLabel.source)
                .join(PairLabel, PairLabel.document_id == Document.id)
                .where(Document.url.like("https://dl.test/%"))
            ).all()
        return {(url, source): matched for url, matched, source in rows}

    def test_listed_concepts_are_positive_and_the_rest_negative(self, setup):
        csv_text = (
            "document_url,concepts\nhttps://dl.test/1,Riot; Strike\nhttps://dl.test/2,none\n"
        )
        with session_scope() as session:
            report = import_document_labels(session, setup, csv_text)
        assert (report["documents"], report["positives"], report["negatives"]) == (2, 2, 4)
        with session_scope() as session:
            labels = session.scalars(select(PairLabel)).all()
            assert len(labels) == 6
            assert all(label.source == "human" for label in labels)

    def test_a_blank_row_is_not_yet_labelled(self, setup):
        with session_scope() as session:
            report = import_document_labels(
                session, setup, "document_url,concepts\nhttps://dl.test/1,\n"
            )
        assert report["skipped_blank"] == 1
        assert report["documents"] == 0

    def test_an_unknown_name_rejects_the_whole_row(self, setup):
        with session_scope() as session:
            report = import_document_labels(
                session, setup, "document_url,concepts\nhttps://dl.test/1,Riot;Volcano\n"
            )
            assert report["documents"] == 0
            assert "Volcano" in report["errors"][0]
            assert session.scalars(select(PairLabel)).all() == []


@pytest.mark.requires_db
def test_predictions_treat_errors_and_absences_as_no():
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-pred", name="Pred")
        concept = service.create_concept(
            session, ontology.id, name="Riot", definition="A riot."
        )
        other = service.create_concept(
            session, ontology.id, name="Flood", definition="A flood."
        )
        document = Document(url="https://pr.test/1", url_hash="prtest000001")
        run = Run(
            name="p",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="c",
            judge_key="j_c",
            status="done",
        )
        session.add_all([document, run])
        session.flush()
        session.add(
            Verdict(
                run_id=run.id,
                document_id=document.id,
                concept_id=concept.id,
                matched=None,
                error="boom",
                samples=1,
            )
        )
        session.flush()
        keys = {(document.id, concept.id), (document.id, other.id)}
        assert article.predictions(session, run.id, keys) == {key: False for key in keys}
        assert article.judged_keys(session, run.id, keys) == set()


@pytest.mark.requires_db
def test_only_a_labelled_prefix_of_the_frozen_order_counts():
    """A document labelled out of turn would break the random-sample guarantee,
    so it is reported, not counted."""
    from hontology.evalkit.arms import labelled_sample
    from hontology.evalkit.labels import upsert_label

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-prefix", name="Prefix")
        concepts = [
            service.create_concept(session, ontology.id, name=n, definition=f"{n}.").id
            for n in ("Riot", "Flood")
        ]
        documents = []
        for i in range(3):
            document = Document(url=f"https://px.test/{i}", url_hash=f"pxtest00000{i}")
            session.add(document)
            session.flush()
            documents.append(document.id)
        for doc_id in (documents[0], documents[2]):
            for concept_id in concepts:
                upsert_label(
                    session,
                    document_id=doc_id,
                    concept_id=concept_id,
                    matched=False,
                    ontology_id=ontology.id,
                )
        manifest = {"order": [{"document_id": d} for d in documents]}
        result = labelled_sample(session, ontology.id, manifest)
    assert result["prefix"] == 1
    assert result["out_of_turn"] == [documents[2]]
    assert set(result["truth"]) == {(documents[0], c) for c in concepts}
