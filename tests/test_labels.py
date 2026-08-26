"""The ground-truth bank: provenance gating and staleness.

Both failure modes here are silent — nothing about a stale label or an unreviewed
machine label looks wrong at the point of use — so each rule is pinned explicitly.
"""

from __future__ import annotations

import pytest

from hontology.db.models import Document, PairLabel
from hontology.db.session import session_scope
from hontology.evalkit import labels
from hontology.ontology import service, snapshots

pytestmark = pytest.mark.requires_db


@pytest.fixture
def bank():
    """An ontology with two concepts and two documents."""

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-labels", name="Labels")
        riot = service.create_concept(
            session, ontology.id, name="Riot", definition="A violent public disturbance."
        )
        strike = service.create_concept(
            session, ontology.id, name="Strike", definition="Workers stop work."
        )
        docs = []
        for i in range(2):
            document = Document(url=f"https://lbl.test/{i}", url_hash=f"lblhash{i:09d}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        ids = {
            "ontology": ontology.id,
            "riot": riot.id,
            "strike": strike.id,
            "docs": docs,
        }
    yield ids


class TestProvenance:
    def test_machine_labels_are_excluded_by_default(self, bank):
        """Scoring an LLM against another LLM's unreviewed labels measures
        agreement between models, not correctness."""
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.MACHINE,
                proposed_by="some-model",
            )

        with session_scope() as session:
            trusted = labels.trusted_labels(session, bank["ontology"])
            assert trusted == []

            with_machine = labels.trusted_labels(
                session, bank["ontology"], include_machine=True
            )
            assert len(with_machine) == 1

    def test_adjudication_promotes_a_machine_label(self, bank):
        with session_scope() as session:
            label = labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.MACHINE,
            )
            label_id = label.id

        with session_scope() as session:
            labels.adjudicate(session, label_id, matched=True, note="confirmed")

        with session_scope() as session:
            assert len(labels.trusted_labels(session, bank["ontology"])) == 1
            assert session.get(PairLabel, label_id).source == labels.ADJUDICATED

    def test_adjudication_can_flip_the_verdict(self, bank):
        """The human disagreeing is the most valuable outcome, not an error."""
        with session_scope() as session:
            label_id = labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.MACHINE,
            ).id

        with session_scope() as session:
            labels.adjudicate(session, label_id, matched=False)

        with session_scope() as session:
            label = session.get(PairLabel, label_id)
            assert label.matched is False
            assert label.source == labels.ADJUDICATED

    def test_human_labels_count_immediately(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=False,
                source=labels.HUMAN,
            )
        with session_scope() as session:
            assert len(labels.trusted_labels(session, bank["ontology"])) == 1

    def test_both_polarities_are_storable(self, bank):
        """A rejected match is the label precision is measured from."""
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )
            labels.upsert_label(
                session, document_id=bank["docs"][1], concept_id=bank["riot"], matched=False
            )
        with session_scope() as session:
            summary = labels.stats(session, bank["ontology"])
            assert summary["positives"] == 1
            assert summary["negatives"] == 1

    def test_relabelling_a_pair_updates_rather_than_duplicates(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=False
            )
        with session_scope() as session:
            summary = labels.stats(session, bank["ontology"])
            assert summary["total"] == 1
            assert summary["negatives"] == 1


class TestStaleness:
    def test_editing_a_definition_makes_its_labels_stale(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )

        with session_scope() as session:
            service.update_concept(
                session, bank["riot"], definition="A violent disturbance involving a crowd."
            )
            snapshots.resolve_current(session, bank["ontology"])

        with session_scope() as session:
            stale = labels.stale_label_ids(session, bank["ontology"])
            assert len(stale) == 1
            # Excluded from the default denominator.
            assert labels.trusted_labels(session, bank["ontology"]) == []
            # Available on request, with the count visible either way.
            assert (
                len(labels.trusted_labels(session, bank["ontology"], include_stale=True)) == 1
            )

    def test_editing_one_concept_does_not_rot_the_others(self, bank):
        """The reason staleness is per concept rather than per version."""
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )
            labels.upsert_label(
                session, document_id=bank["docs"][1], concept_id=bank["strike"], matched=True
            )

        with session_scope() as session:
            service.update_concept(session, bank["riot"], definition="Reworded entirely.")
            snapshots.resolve_current(session, bank["ontology"])

        with session_scope() as session:
            surviving = labels.trusted_labels(session, bank["ontology"])
            assert len(surviving) == 1
            assert surviving[0].concept_id == bank["strike"]

    def test_a_weight_change_does_not_make_labels_stale(self, bank):
        """Retuning scoring must never invalidate ground truth."""
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )

        with session_scope() as session:
            service.update_concept(session, bank["riot"], weight=42.0)
            snapshots.resolve_current(session, bank["ontology"])

        with session_scope() as session:
            assert labels.stale_label_ids(session, bank["ontology"]) == set()
            assert len(labels.trusted_labels(session, bank["ontology"])) == 1

    def test_re_adjudicating_clears_staleness(self, bank):
        """A human re-reading against the new wording makes the label valid again."""
        with session_scope() as session:
            label_id = labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            ).id

        with session_scope() as session:
            service.update_concept(session, bank["riot"], definition="Reworded again.")
            snapshots.resolve_current(session, bank["ontology"])

        with session_scope() as session:
            assert len(labels.stale_label_ids(session, bank["ontology"])) == 1
            labels.adjudicate(session, label_id, matched=True)

        with session_scope() as session:
            assert labels.stale_label_ids(session, bank["ontology"]) == set()
            assert len(labels.trusted_labels(session, bank["ontology"])) == 1

    def test_stats_surface_the_stale_count(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session, document_id=bank["docs"][0], concept_id=bank["riot"], matched=True
            )
        with session_scope() as session:
            service.update_concept(session, bank["riot"], definition="Different wording now.")
            snapshots.resolve_current(session, bank["ontology"])
        with session_scope() as session:
            assert labels.stats(session, bank["ontology"])["stale"] == 1
