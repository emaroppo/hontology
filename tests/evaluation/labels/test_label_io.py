"""Moving the ground-truth bank in and out.

The bank is hours of human attention, so the rules that stop an import from
damaging it are pinned here alongside the round-trip.
"""

from __future__ import annotations

import csv
import io

import pytest
from sqlalchemy import delete, select

from hontology.db.models import Document, Observation, PairLabel
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank as labels
from hontology.evaluation.labels import csv_export, csv_io, staleness
from hontology.ontology import service, snapshots

pytestmark = pytest.mark.requires_db


@pytest.fixture
def bank():

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-io", name="IO")
        riot = service.create_concept(
            session, ontology.id, name="Riot", definition="A violent disturbance."
        )
        strike = service.create_concept(
            session, ontology.id, name="Strike", definition="Workers stop work."
        )
        docs = []
        for i in range(3):
            document = Document(url=f"https://io.test/{i}", url_hash=f"iohash{i:010d}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        # A second ontology proves the export is scoped.
        other = service.create_ontology(session, slug="test-io-other", name="Other")
        service.create_concept(session, other.id, name="Riot", definition="Elsewhere.")

        ids = {
            "ontology": ontology.id,
            "other": other.id,
            "riot": riot.id,
            "strike": strike.id,
            "docs": docs,
        }
    yield ids


def rows_of(csv_text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def labels_of(session, bank) -> list[PairLabel]:
    """This fixture's labels only.

    A bare ``select(PairLabel)`` reads every row in the database, so in a shared
    development database a test would assert against somebody else's data.
    """
    return list(
        session.scalars(
            select(PairLabel).where(PairLabel.concept_id.in_([bank["riot"], bank["strike"]]))
        )
    )


def one_label(session, bank) -> PairLabel:
    rows = labels_of(session, bank)
    assert len(rows) == 1, f"expected exactly one label, found {len(rows)}"
    return rows[0]


def clear_labels(session, bank) -> None:
    """Remove only THIS fixture's labels.

    An unqualified ``DELETE FROM pair_labels`` in a test empties the table for
    the whole database — which, in a shared development database, silently
    destroys real work.
    """
    session.execute(
        delete(PairLabel).where(PairLabel.concept_id.in_([bank["riot"], bank["strike"]]))
    )
    session.flush()


class TestExport:
    def test_header_and_content(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.HUMAN,
                note="clear case",
            )

        with session_scope() as session:
            exported = csv_export.export_labels(session, bank["ontology"])

        rows = rows_of(exported)
        assert len(rows) == 1
        assert rows[0]["document_url"] == "https://io.test/0"
        assert rows[0]["concept"] == "Riot"
        assert rows[0]["matched"] == "true"
        assert rows[0]["source"] == "human"
        assert rows[0]["note"] == "clear case"

    def test_export_is_scoped_to_one_ontology(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
            )
        with session_scope() as session:
            assert rows_of(csv_export.export_labels(session, bank["other"])) == []

    def test_stale_labels_are_flagged_not_dropped(self, bank):
        """Omitting them would make an export smaller than the work behind it."""
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
            )
        with session_scope() as session:
            service.update_concept(session, bank["riot"], definition="Reworded.")
            snapshots.resolve_current(session, bank["ontology"])

        with session_scope() as session:
            rows = rows_of(csv_export.export_labels(session, bank["ontology"]))
        assert len(rows) == 1
        assert rows[0]["stale"] == "true"

    def test_empty_bank_exports_a_header_only(self, bank):
        with session_scope() as session:
            exported = csv_export.export_labels(session, bank["ontology"])
        assert exported.strip() == ",".join(csv_export.LABEL_COLUMNS)


class TestRoundTrip:
    def test_labels_survive_export_and_reimport(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.HUMAN,
            )
            labels.upsert_label(
                session,
                document_id=bank["docs"][1],
                concept_id=bank["strike"],
                matched=False,
                source=labels.ADJUDICATED,
            )

        with session_scope() as session:
            exported = csv_export.export_labels(session, bank["ontology"])
            clear_labels(session, bank)

        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], exported)
            assert report["created"] == 2

        with session_scope() as session:
            restored = {
                (label.concept_id, label.matched, label.source)
                for label in labels_of(session, bank)
            }
        assert (bank["riot"], True, "human") in restored
        assert (bank["strike"], False, "adjudicated") in restored

    def test_the_version_stamp_is_preserved_not_restamped(self, bank):
        """Re-stamping would claim a human read today's wording when they did not,
        which quietly defeats staleness detection."""
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
            )
        with session_scope() as session:
            original = one_label(session, bank).ontology_version
            exported = csv_export.export_labels(session, bank["ontology"])
            clear_labels(session, bank)

        # The ontology moves on before the import.
        with session_scope() as session:
            service.update_concept(session, bank["riot"], definition="Different now.")
            new_version = snapshots.resolve_current(session, bank["ontology"]).version
        assert new_version != original

        with session_scope() as session:
            csv_io.import_labels(session, bank["ontology"], exported)

        with session_scope() as session:
            label = one_label(session, bank)
            assert label.ontology_version == original
            # And it is therefore correctly recognised as stale.
            assert label.id in staleness.stale_label_ids(session, bank["ontology"])


class TestImportSafety:
    def test_existing_labels_are_not_overwritten_by_default(self, bank):
        """An import must never silently destroy adjudicated work."""
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.ADJUDICATED,
            )

        incoming = (
            "document_url,concept,matched,source\nhttps://io.test/0,Riot,false,imported\n"
        )
        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], incoming)

        assert report["skipped_existing"] == 1
        assert report["created"] == 0
        with session_scope() as session:
            label = one_label(session, bank)
            assert label.matched is True
            assert label.source == "adjudicated"

    def test_overwrite_is_opt_in_and_reported(self, bank):
        with session_scope() as session:
            labels.upsert_label(
                session,
                document_id=bank["docs"][0],
                concept_id=bank["riot"],
                matched=True,
                source=labels.ADJUDICATED,
            )

        incoming = (
            "document_url,concept,matched,source\nhttps://io.test/0,Riot,false,imported\n"
        )
        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], incoming, overwrite=True)
        assert report["updated"] == 1
        with session_scope() as session:
            assert one_label(session, bank).matched is False

    def test_unknown_documents_become_stubs(self, bank):
        """Labels legitimately arrive before the corpus does."""
        incoming = "document_url,concept,matched\nhttps://io.test/brand-new,Riot,true\n"
        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], incoming)

        assert report["documents_created"] == 1
        assert report["created"] == 1
        with session_scope() as session:
            document = session.scalar(
                select(Document).where(Document.url == "https://io.test/brand-new")
            )
            assert document is not None
            assert document.body_path is None  # a stub; the scraper fills it later

    def test_stub_creation_can_be_refused(self, bank):
        incoming = "document_url,concept,matched\nhttps://io.test/nope,Riot,true\n"
        with session_scope() as session:
            report = csv_io.import_labels(
                session, bank["ontology"], incoming, create_missing_documents=False
            )
        assert report["created"] == 0
        assert report["bad_row_count"] == 1

    def test_unknown_concepts_are_reported_not_invented(self, bank):
        incoming = (
            "document_url,concept,matched\n"
            "https://io.test/0,Flood,true\n"
            "https://io.test/1,Riot,true\n"
        )
        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], incoming)

        assert report["unknown_concepts"] == ["Flood"]
        assert report["created"] == 1  # the Riot row still landed

    def test_unreadable_rows_are_collected_not_fatal(self, bank):
        """One bad line must not cost the rest of the file."""
        incoming = (
            "document_url,concept,matched\n"
            "https://io.test/0,Riot,perhaps\n"
            ",Riot,true\n"
            "https://io.test/1,Riot,true\n"
        )
        with session_scope() as session:
            report = csv_io.import_labels(session, bank["ontology"], incoming)

        assert report["created"] == 1
        assert report["bad_row_count"] == 2

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_common_boolean_spellings_are_accepted(self, bank, raw, expected):
        """People edit these files in spreadsheets."""
        incoming = f"document_url,concept,matched\nhttps://io.test/0,Riot,{raw}\n"
        with session_scope() as session:
            csv_io.import_labels(session, bank["ontology"], incoming, overwrite=True)
        with session_scope() as session:
            assert one_label(session, bank).matched is expected


class TestObservations:
    def test_round_trip(self, bank):
        incoming = (
            "concept,locus_iso3,occurred_on,description\n"
            "Riot,KEN,2026-08-01,A riot in Nairobi\n"
        )
        with session_scope() as session:
            report = csv_io.import_observations(session, bank["ontology"], incoming)
        assert report["created"] == 1

        with session_scope() as session:
            exported = rows_of(csv_export.export_observations(session, bank["ontology"]))
        assert len(exported) == 1
        assert exported[0]["locus_iso3"] == "KEN"
        assert exported[0]["occurred_on"] == "2026-08-01"

    def test_reimport_is_idempotent(self, bank):
        incoming = "concept,locus_iso3,occurred_on\nRiot,KEN,2026-08-01\n"
        with session_scope() as session:
            csv_io.import_observations(session, bank["ontology"], incoming)
        with session_scope() as session:
            report = csv_io.import_observations(session, bank["ontology"], incoming)
        assert report["created"] == 0
        assert report["updated"] == 1
        with session_scope() as session:
            mine = list(
                session.scalars(
                    select(Observation).where(Observation.concept_id == bank["riot"])
                )
            )
            assert len(mine) == 1

    def test_an_unknown_locus_is_reported(self, bank):
        incoming = "concept,locus_iso3,occurred_on\nRiot,ZZZ,2026-08-01\n"
        with session_scope() as session:
            report = csv_io.import_observations(session, bank["ontology"], incoming)
        assert report["created"] == 0
        assert report["bad_row_count"] == 1
