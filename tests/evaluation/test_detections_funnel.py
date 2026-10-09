"""Detection export and the pipeline funnel.

Both are usable with no ground truth at all, which is the point of them: they
answer "what did this produce" and "where did the volume go" before anyone has
labelled anything.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from hontology.db.models import Candidate, Document, FeedEvent, FeedSlice, Locus, Run, Verdict
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank
from hontology.evaluation.outputs import detections as det
from hontology.evaluation.outputs import funnel as funnel_module
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def run_with_output():
    """A run with two matches, one of which a human has confirmed."""
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-det", name="Detections")
        riot = service.create_concept(session, ontology.id, name="Riot", definition="A riot.")
        strike = service.create_concept(
            session, ontology.id, name="Strike", definition="A strike."
        )
        kenya = session.scalar(select(Locus).where(Locus.iso3 == "KEN"))
        locus_id = kenya.id

        feed_slice = FeedSlice(
            feed="test_det",
            slice_key="20260827000000",
            sliced_at=datetime.now(UTC),
            status="ok",
        )
        session.add(feed_slice)
        session.flush()

        docs = []
        for i in range(4):
            document = Document(
                url=f"https://det.test/{i}",
                url_hash=f"dethash{i:09d}",
                body_path=f"{i}.txt",
                fetched_at=datetime.now(UTC),
            )
            session.add(document)
            session.flush()
            docs.append(document.id)
            session.add(
                FeedEvent(
                    slice_id=feed_slice.id,
                    feed_event_id=f"fe{i}",
                    document_id=document.id,
                    occurred_on="20260826",
                )
            )

        run = Run(
            name="det",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="ck",
            judge_key="jk_ck",
            status="done",
        )
        session.add(run)
        session.flush()

        for document_id in docs:
            for concept_id in (riot.id, strike.id):
                session.add(
                    Candidate(
                        run_id=run.id,
                        document_id=document_id,
                        concept_id=concept_id,
                        source="semantic",
                        score=0.6,
                        rank=1,
                        selected=concept_id == riot.id,
                    )
                )

        # Two documents match Riot; the other two do not.
        for index, document_id in enumerate(docs):
            session.add(
                Verdict(
                    run_id=run.id,
                    document_id=document_id,
                    concept_id=riot.id,
                    matched=index < 2,
                    confidence=0.9 if index < 2 else 0.2,
                    locus_id=locus_id if index < 2 else None,
                    evidence="a crowd overturned cars" if index < 2 else None,
                    model="test-model",
                    prompt_id="strict_v1",
                )
            )

        ids = {
            "ontology": ontology.id,
            "riot": riot.id,
            "run": run.id,
            "docs": docs,
            "locus": locus_id,
        }

    # A human confirms the first detection only.
    with session_scope() as session:
        bank.upsert_label(
            session,
            document_id=ids["docs"][0],
            concept_id=ids["riot"],
            matched=True,
            source=bank.HUMAN,
        )
    return ids


def rows_of(csv_text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(csv_text)))


class TestDetections:
    def test_only_matches_are_exported(self, run_with_output):
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"])
        assert len(rows) == 2
        assert all(r.concept == "Riot" for r in rows)

    def test_verification_distinguishes_confirmed_from_unreviewed(self, run_with_output):
        """A detection is a claim, not a fact; exporting them alike would launder
        model output into apparent ground truth."""
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"])
        statuses = sorted(r.verification for r in rows)
        assert statuses == [det.CONFIRMED, det.UNVERIFIED]

    def test_the_date_comes_from_the_feed_not_the_scrape(self, run_with_output):
        """A verdict has no date — the model is asked whether, not when."""
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"])
        assert all(r.occurred_on == "2026-08-26" for r in rows)

    def test_locus_is_emitted_in_both_code_systems(self, run_with_output):
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"])
        assert all(r.locus_iso3 == "KEN" for r in rows)
        assert all(r.locus_iso2 == "KE" for r in rows)

    def test_evidence_travels_with_the_claim(self, run_with_output):
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"])
        assert all(r.evidence for r in rows)

    def test_verified_only_filters_to_confirmed(self, run_with_output):
        with session_scope() as session:
            rows = det.detections(session, run_with_output["run"], verified_only=True)
        assert len(rows) == 1
        assert rows[0].verification == det.CONFIRMED

    def test_min_confidence_filters(self, run_with_output):
        with session_scope() as session:
            assert det.detections(session, run_with_output["run"], min_confidence=0.95) == []

    def test_csv_has_the_documented_columns(self, run_with_output):
        with session_scope() as session:
            rows = rows_of(det.export_detections(session, run_with_output["run"]))
        assert list(rows[0]) == det.DETECTION_COLUMNS

    def test_a_missing_run_is_an_error(self, run_with_output):
        with session_scope() as session, pytest.raises(LookupError):
            det.detections(session, 987654)


class TestEvents:
    def test_documents_collapse_to_one_event(self, run_with_output):
        """Two outlets reporting one riot is one fact, not two."""
        with session_scope() as session:
            rows = det.events(session, run_with_output["run"])
        assert len(rows) == 1
        assert len(rows[0].document_ids) == 2

    def test_one_confirmation_confirms_the_event(self, run_with_output):
        with session_scope() as session:
            rows = det.events(session, run_with_output["run"])
        assert rows[0].verification == det.CONFIRMED
        assert rows[0].confirmed == 1

    def test_an_event_is_rejected_only_when_every_document_is(self, run_with_output):
        """One bad article does not disprove an event the others evidence."""
        event = det.Event(run_id=1, concept="Riot", locus_iso3="KEN", occurred_on="2026-08-26")
        event.document_ids = {1, 2}
        event.rejected = 1
        assert event.verification == det.UNVERIFIED
        event.rejected = 2
        assert event.verification == det.REJECTED

    def test_summary_counts_by_status(self, run_with_output):
        with session_scope() as session:
            stats = det.summary(session, run_with_output["run"])
        assert stats["detections"] == 2
        assert stats["events"] == 1
        assert stats["by_verification"] == {det.CONFIRMED: 1, det.UNVERIFIED: 1}


class TestFunnel:
    def test_it_works_with_no_labels_at_all(self, run_with_output):
        """The whole point: usable before anyone has labelled anything."""
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        assert result["pairs"][0]["count"] == 8  # pool
        assert result["pairs"][-1]["count"] == 2  # matched

    def test_each_step_reports_its_share_of_the_previous(self, run_with_output):
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        selected = result["pairs"][1]
        assert selected["count"] == 4
        assert selected["of_previous"] == pytest.approx(0.5)

    def test_the_first_step_has_no_previous_share(self, run_with_output):
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        assert result["documents"][0]["of_previous"] is None
        assert result["pairs"][0]["of_previous"] is None

    def test_unjudged_selected_pairs_are_surfaced(self, run_with_output):
        """A run that stopped early looks identical to a working one otherwise."""
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        # 4 selected, 4 judged, so nothing outstanding.
        assert result["unjudged_selected"] == 0

    def test_errored_verdicts_are_counted_separately(self, run_with_output):
        with session_scope() as session:
            verdict = session.scalar(
                select(Verdict).where(Verdict.run_id == run_with_output["run"])
            )
            verdict.matched = None
            verdict.error = "boom"
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        assert result["errored_verdicts"] == 1

    def test_every_step_explains_what_a_healthy_drop_looks_like(self, run_with_output):
        """Attrition is mostly the pipeline working, not failing."""
        with session_scope() as session:
            result = funnel_module.funnel(session, run_with_output["run"])
        assert all(step["note"] for step in result["documents"])
        assert all(step["note"] for step in result["pairs"])

    def test_a_missing_run_is_an_error(self, run_with_output):
        with session_scope() as session, pytest.raises(LookupError):
            funnel_module.funnel(session, 987654)
