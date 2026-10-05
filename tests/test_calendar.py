"""Event calendars: loading, windows, and stage-by-stage scoring.

The scoring tests build one small world by hand: a run, a few documents placed
in and out of a window, and verdicts on some of them. Each entry's stage counts
then say exactly where it was kept or lost.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hontology.db.models import (
    Candidate,
    Document,
    FeedArticle,
    FeedEvent,
    FeedSlice,
    Run,
    Verdict,
)
from hontology.db.session import session_scope
from hontology.evalkit import calendar
from hontology.ingest.loci import by_fips
from hontology.ontology import service

HEADER = "id,kind,concept,country_iso3,date,end_date,precursor_of,description,sources,notes\n"


def write(tmp_path, *rows: str):
    path = tmp_path / "calendar.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestLoad:
    def test_rows_load_with_alternative_countries(self, tmp_path):
        path = write(
            tmp_path,
            "pipe,positive,Pipeline shutdown,HUN|SVK,2025-08-18,,,,,",
            "warn,precursor,Energy supply warning,HUN,2025-08-15,,pipe,,,",
        )
        entries = {e.id: e for e in calendar.load(path)}
        assert entries["pipe"].countries == ("HUN", "SVK")
        assert entries["warn"].precursor_of == "pipe"
        assert entries["pipe"].date == date(2025, 8, 18)

    @pytest.mark.parametrize(
        "row, message",
        [
            ("a,rumour,Port closure,HKG,2025-01-01,,,,,", "kind"),
            ("a,positive,Port closure,,2025-01-01,,,,,", "no country"),
            ("a,positive,Port closure,HKG,01/01/2025,,,,,", "bad date"),
            ("a,precursor,Port closure,HKG,2025-01-01,,nowhere,,,", "points nowhere"),
        ],
    )
    def test_bad_rows_are_refused(self, tmp_path, row, message):
        with pytest.raises(calendar.CalendarError, match=message):
            calendar.load(write(tmp_path, row))

    def test_duplicate_ids_are_refused(self, tmp_path):
        row = "a,positive,Port closure,HKG,2025-01-01,,,,,"
        with pytest.raises(calendar.CalendarError, match="duplicate"):
            calendar.load(write(tmp_path, row, row))


class TestWindows:
    def test_window_is_whole_days_around_the_date(self):
        entry = calendar.Entry("a", "positive", "X", ("HKG",), date(2025, 9, 23), None, "")
        start, end = entry.window(1, 2)
        assert start == datetime(2025, 9, 22, tzinfo=UTC)
        assert end == datetime(2025, 9, 26, tzinfo=UTC)

    def test_days_merge_places_across_entries(self):
        a = calendar.Entry("a", "positive", "X", ("HKG",), date(2025, 9, 23), None, "")
        b = calendar.Entry("b", "positive", "X", ("JPN",), date(2025, 9, 24), None, "")
        days = calendar.days_to_ingest([a, b], {"HKG": 1, "JPN": 2}, before=0, after=1)
        assert days == {
            date(2025, 9, 23): [1],
            date(2025, 9, 24): [1, 2],
            date(2025, 9, 25): [2],
        }


@pytest.mark.requires_db
class TestEvaluate:
    """One run, one concept, a window in Hong Kong around 2025-09-23."""

    @pytest.fixture
    def world(self, tmp_path):
        with session_scope() as session:
            ontology = service.create_ontology(session, slug="test-calendar", name="Cal")
            closure = service.create_concept(
                session, ontology.id, name="Port closure", definition="A port stops."
            )
            warning = service.create_concept(
                session, ontology.id, name="Storm warning", definition="A warning."
            )
            run = Run(
                name="cal",
                ontology_id=ontology.id,
                ontology_version="v1",
                config={},
                candidates_key="ck",
                judge_key="jk_ck",
                status="done",
            )
            session.add(run)
            hk = by_fips(session)["HK"].id
            us = by_fips(session)["US"].id
            feed_slice = FeedSlice(
                feed="test_calendar",
                slice_key="20250923000000",
                sliced_at=datetime(2025, 9, 23, tzinfo=UTC),
                status="ok",
            )
            session.add(feed_slice)
            session.flush()

            def document(name: str, *, fetched: bool) -> int:
                doc = Document(
                    url=f"https://cal.test/{name}",
                    url_hash=f"cal{name:>013}",
                    body_path=f"{name}.txt" if fetched else None,
                )
                session.add(doc)
                session.flush()
                return doc.id

            # In the window, via the event export: fetched, retrieved, matched.
            hit = document("hit", fetched=True)
            session.add(
                FeedEvent(
                    slice_id=feed_slice.id, feed_event_id="e1", document_id=hit, locus_id=hk
                )
            )
            # In the window, via the knowledge graph only: fetched, retrieved, rejected.
            seen = document("seen", fetched=True)
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id="g1",
                    document_id=seen,
                    published_at=datetime(2025, 9, 22, 6, tzinfo=UTC),
                    themes=["MARITIME"],
                    locus_ids=[hk],
                )
            )
            # In the window but never fetched.
            dead = document("dead", fetched=False)
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id="g2",
                    document_id=dead,
                    published_at=datetime(2025, 9, 23, 6, tzinfo=UTC),
                    themes=[],
                    locus_ids=[hk],
                )
            )
            # Matched, but in the wrong country: must not count.
            away = document("away", fetched=True)
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id="g3",
                    document_id=away,
                    published_at=datetime(2025, 9, 23, 6, tzinfo=UTC),
                    themes=[],
                    locus_ids=[us],
                )
            )
            session.flush()

            for doc_id in (hit, seen, away):
                for concept_id in (closure.id, warning.id):
                    session.add(
                        Candidate(
                            run_id=run.id,
                            document_id=doc_id,
                            concept_id=concept_id,
                            source="semantic",
                            score=0.9,
                            rank=1,
                            selected=True,
                        )
                    )
            for doc_id, concept_id, matched in (
                (hit, closure.id, True),
                (seen, closure.id, False),
                (away, closure.id, True),
                (seen, warning.id, True),
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
            run_id = run.id

        path = write(
            tmp_path,
            "closure,positive,Port closure,HKG,2025-09-23,,,,,",
            "storm,precursor,Storm warning,HKG,2025-09-23,,closure,,,",
            "quiet,control,Port closure,HKG,2025-12-17,,,,,",
            "elsewhere,positive,Port closure,JPN,2025-09-23,,,,,",
        )
        return run_id, calendar.load(path)

    def _result(self, world):
        run_id, entries = world
        with session_scope() as session:
            return calendar.evaluate(session, run_id, entries)

    def test_stages_count_each_documents_progress(self, world):
        rows = {r["id"]: r for r in self._result(world)["entries"]}
        assert {s: rows["closure"][s] for s in calendar.STAGES} == {
            "in_feed": 3,
            "passed_filter": 3,
            "fetched": 2,
            "retrieved": 2,
            "matched": 1,
        }
        assert rows["closure"]["detected"] is True

    def test_a_match_in_another_country_does_not_count(self, world):
        rows = {r["id"]: r for r in self._result(world)["entries"]}
        assert rows["elsewhere"]["detected"] is False
        assert rows["elsewhere"]["lost_at"] == "in_feed"

    def test_a_quiet_control_is_not_a_false_alarm(self, world):
        summary = self._result(world)["summary"]
        assert summary["false_alarm_rate"]["hits"] == 0
        assert summary["event_recall"] == summary["event_recall"] | {"hits": 1, "n": 2}

    def test_lead_time_runs_from_first_sighting_to_the_disruption_day(self, world):
        """The warning was matched on an article first seen 18 hours before."""
        (lead,) = self._result(world)["lead_times"]
        assert lead["precursor"] == "storm"
        assert lead["lead_days"] == 0.75
        assert lead["disruption_detected"] is True

    def test_unknown_concepts_are_refused(self, world, tmp_path):
        run_id, _ = world
        entries = calendar.load(write(tmp_path, "x,positive,Volcano,HKG,2025-09-23,,,,,"))
        with session_scope() as session, pytest.raises(calendar.CalendarError, match="Volcano"):
            calendar.evaluate(session, run_id, entries)
