"""The labelled sample's frame: only documents the run actually processed."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from hontology.db.models import Candidate, Document, FeedArticle, FeedSlice, Locus, Run
from hontology.db.session import session_scope
from hontology.evalkit.calendar import Entry
from hontology.evalkit.sample import frame, manifest
from hontology.ontology import service

pytestmark = pytest.mark.requires_db

ENTRY = Entry("e", "positive", "Port closure", ("HKG",), date(2025, 9, 23), None, "")


@pytest.fixture
def world() -> dict:
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-frame", name="Frame")
        concept = service.create_concept(
            session, ontology.id, name="Port closure", definition="A port stops."
        )
        run = Run(
            name="frame",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="fk",
            judge_key="fj_fk",
            status="running",
        )
        hk = session.scalar(select(Locus.id).where(Locus.iso3 == "HKG"))
        feed_slice = FeedSlice(
            feed="test_frame",
            slice_key="20250923000000",
            sliced_at=datetime(2025, 9, 23, tzinfo=UTC),
            status="ok",
        )
        session.add_all([run, feed_slice])
        session.flush()
        ids = {}
        for name in ("processed", "late"):
            doc = Document(
                url=f"https://frame.test/{name}",
                url_hash=f"frame{name:>011}",
                body_path=f"{name}.txt",
            )
            session.add(doc)
            session.flush()
            ids[name] = doc.id
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id=f"r-{name}",
                    document_id=doc.id,
                    published_at=datetime(2025, 9, 23, 6, tzinfo=UTC),
                    themes=[],
                    locus_ids=[hk],
                )
            )
        # Retrieval stores the full pool, so a processed document has rows even
        # when nothing was selected for it.
        session.add(
            Candidate(
                run_id=run.id,
                document_id=ids["processed"],
                concept_id=concept.id,
                source="semantic",
                score=0.42,
                rank=1,
                selected=False,
            )
        )
        session.flush()
        return {"run": run.id, **ids}


def test_a_document_the_run_never_processed_is_left_out_and_counted(world):
    with session_scope() as session:
        documents, late = frame(session, world["run"], [ENTRY], before=1, after=2)
    assert [d.document_id for d in documents] == [world["processed"]]
    assert late == 1


def test_a_processed_document_with_nothing_selected_stays_in(world):
    """Leaving these out would hide what retrieval missed from recall."""
    with session_scope() as session:
        documents, _ = frame(session, world["run"], [ENTRY], before=1, after=2)
    assert documents[0].band == "low"


def test_the_manifest_records_what_was_left_out(world):
    with session_scope() as session:
        documents, late = frame(session, world["run"], [ENTRY], before=1, after=2)
    record = manifest(world["run"], "abc", 1, documents, late_arrivals=late)
    assert record["left_out_late_arrivals"] == 1
    assert record["size"] == 1
