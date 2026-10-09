"""Article-level scoring, the frozen sample order, and whole-document labels."""

from __future__ import annotations

from collections import Counter

import pytest
from sqlalchemy import select

from hontology.db.models import Document, PairLabel, Run, Verdict
from hontology.db.session import session_scope
from hontology.evaluation import article
from hontology.evaluation.labels import sample
from hontology.evaluation.labels.document_labels import import_document_labels
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
    from hontology.evaluation.comparison.arms import labelled_sample
    from hontology.evaluation.labels.bank import upsert_label

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


class TestEqualPerWindow:
    """A crisis window with thousands of articles must not fill the sample."""

    FRAME = docs(90, "big", "high") + docs(10, "small", "high", 500)

    def test_every_prefix_shares_itself_equally_between_windows(self):
        ordered = sample.frozen_order(self.FRAME, seed=4, allocation=sample.EQUAL_PER_WINDOW)
        for size in (2, 10, 20):
            counts = Counter(d.window for d in ordered[:size])
            assert counts["big"] == counts["small"] == size // 2

    def test_a_small_window_running_out_leaves_the_rest_to_the_others(self):
        ordered = sample.frozen_order(self.FRAME, seed=4, allocation=sample.EQUAL_PER_WINDOW)
        assert sorted(d.document_id for d in ordered) == sorted(
            d.document_id for d in self.FRAME
        )
        assert Counter(d.window for d in ordered[20:]) == {"big": 80}

    def test_the_manifest_records_allocation_and_window_sizes(self):
        ordered = sample.frozen_order(self.FRAME, seed=4, allocation=sample.EQUAL_PER_WINDOW)
        record = sample.manifest(1, "abc", 4, ordered, allocation=sample.EQUAL_PER_WINDOW)
        assert record["allocation"] == sample.EQUAL_PER_WINDOW
        assert record["window_sizes"] == {"big": 90, "small": 10}

    def test_a_document_counts_for_its_window_in_the_frame(self):
        """Two labelled from a window of 90 count 45 each; two of 10 count 5."""
        ordered = sample.frozen_order(self.FRAME, seed=4, allocation=sample.EQUAL_PER_WINDOW)
        record = sample.manifest(1, "abc", 4, ordered, allocation=sample.EQUAL_PER_WINDOW)
        labelled = {d.document_id for d in ordered[:4]}
        weights, groups = article.window_weights(record, labelled)
        assert sorted(weights.values()) == [5.0, 5.0, 45.0, 45.0]
        assert set(groups.values()) == {"big", "small"}

    def test_weighted_scores_estimate_the_frame_not_the_sample(self):
        """The big window's document is right, the small one's wrong: unweighted
        precision is a half, weighted it follows the big window."""
        truth = {(1, 10): True, (2, 10): False}
        predicted = {(1, 10): True, (2, 10): True}
        plain = article.document_bootstrap(truth, predicted, n_boot=50)
        weighted = article.document_bootstrap(
            truth,
            predicted,
            n_boot=50,
            weights={1: 9.0, 2: 1.0},
            groups={1: "big", 2: "small"},
        )
        assert plain["precision"] == 0.5
        assert weighted["precision"] == 0.9
        assert weighted["weighted"] is True
        assert weighted["tp"] == 1 and weighted["fp"] == 1  # counts stay unweighted

    def test_resampling_within_windows_keeps_each_windows_count(self):
        import random

        documents = [1, 2, 3, 4]
        groups = {1: "a", 2: "a", 3: "b", 4: "b"}
        for seed in range(5):
            draw = article._resample(documents, groups, random.Random(seed))
            assert Counter(groups[d] for d in draw) == {"a": 2, "b": 2}

    def test_windows_with_one_labelled_document_still_vary(self):
        """Early in labelling most windows hold one labelled document; resampling
        each alone would redraw it every time and give a zero-width interval."""
        truth = {(d, 10): d % 2 == 0 for d in range(1, 9)}
        predicted = {(d, 10): True for d in range(1, 9)}
        result = article.document_bootstrap(
            truth,
            predicted,
            n_boot=200,
            weights=dict.fromkeys(range(1, 9), 1.0),
            groups={d: f"window-{d}" for d in range(1, 9)},
        )
        low, high = result["precision_ci"]
        assert low < result["precision"] < high

    def test_windows_with_several_labelled_documents_stay_apart(self):
        import random

        documents = [1, 2, 3, 4, 5]
        groups = {1: "a", 2: "a", 3: "b", 4: "c", 5: "d"}
        for seed in range(5):
            draw = article._resample(documents, groups, random.Random(seed))
            assert sum(1 for d in draw if groups[d] == "a") == 2
            assert len(draw) == 5


@pytest.mark.requires_db
def test_one_run_is_scored_on_the_sample_with_its_mistakes():
    """A run judged alone gets the scores the arms report gives it, and every pair
    it got wrong, in the sample's order."""
    from hontology.evaluation.comparison.arms import run_on_sample
    from hontology.evaluation.labels.bank import upsert_label

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-one-run", name="One run")
        riot, flood = (
            service.create_concept(session, ontology.id, name=n, definition=f"{n}.").id
            for n in ("Riot", "Flood")
        )
        run = Run(
            name="r",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="c",
            judge_key="j_c",
            status="done",
        )
        documents = [
            Document(url=f"https://or.test/{i}", url_hash=f"ortest00000{i}") for i in range(2)
        ]
        session.add_all([run, *documents])
        session.flush()
        first, second = (d.id for d in documents)
        truth = {
            (first, riot): True,
            (first, flood): False,
            (second, riot): False,
            (second, flood): True,
        }
        for (doc_id, concept_id), matched in truth.items():
            upsert_label(
                session,
                document_id=doc_id,
                concept_id=concept_id,
                matched=matched,
                ontology_id=ontology.id,
            )
        # Right on the first document; on the second, Riot wrongly yes and Flood never judged.
        for doc_id, concept_id, matched in (
            (first, riot, True),
            (first, flood, False),
            (second, riot, True),
        ):
            session.add(
                Verdict(
                    run_id=run.id,
                    document_id=doc_id,
                    concept_id=concept_id,
                    matched=matched,
                    samples=1,
                )
            )
        session.flush()
        manifest = {"order": [{"document_id": second}, {"document_id": first}]}
        result = run_on_sample(session, run.id, manifest)

    scores = result["article"]["end_to_end"]
    assert (scores["tp"], scores["fp"], scores["fn"]) == (1, 1, 1)
    assert result["article"]["pairs_judged"] == 3
    assert [(e["position"], e["concept_id"], e["kind"]) for e in result["errors"]] == [
        (1, riot, "false positive"),
        (1, flood, "false negative"),
    ]
