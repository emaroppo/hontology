"""Event calendars: loading, windows, and stage-by-stage scoring.

The scoring tests build one small world by hand: a run, a few documents placed
in and out of a window, and verdicts on some of them. Each entry's stage counts
then say exactly where it was kept or lost.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from hontology.db.models import (
    CalendarReview,
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
            "judged": 2,
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

    def test_a_copy_reads_its_representatives_verdict(self, world):
        """A Japanese outlet republishing the US article: the copy is never
        judged itself, but its window counts the representative's match."""
        run_id, entries = world
        with session_scope() as session:
            away = session.scalar(
                select(Document.id).where(Document.url == "https://cal.test/away")
            )
            jp = by_fips(session)["JA"].id
            feed_slice = session.scalar(
                select(FeedSlice).where(FeedSlice.feed == "test_calendar")
            )
            copy = Document(
                url="https://cal.test/copy",
                url_hash="calcopy000000",
                body_path="copy.txt",
                duplicate_of=away,
            )
            session.add(copy)
            session.flush()
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id="g4",
                    document_id=copy.id,
                    published_at=datetime(2025, 9, 23, 9, tzinfo=UTC),
                    themes=[],
                    locus_ids=[jp],
                )
            )
        rows = {r["id"]: r for r in self._result(world)["entries"]}
        assert rows["elsewhere"]["detected"] is True
        assert rows["elsewhere"]["matched"] == 1

    def test_a_retrieved_document_never_judged_is_lost_at_judged(self, world):
        """A budget that stops before a document must not read as a rejection."""
        run_id, entries = world
        with session_scope() as session:
            for verdict in session.scalars(select(Verdict).where(Verdict.run_id == run_id)):
                session.delete(verdict)
        rows = {r["id"]: r for r in self._result(world)["entries"]}
        assert rows["closure"]["retrieved"] == 2
        assert rows["closure"]["lost_at"] == "judged"

    def _set_run(self, run_id, **fields):
        with session_scope() as session:
            run = session.get(Run, run_id)
            for key, value in fields.items():
                setattr(run, key, value)

    def test_only_entries_the_run_finished_are_scored(self, world):
        """An entry the run never reached has no verdicts; scored, it would read
        as a miss (or, for a control, as quiet)."""
        run_id, _ = world
        self._set_run(
            run_id, status="running", manifest={"calendar_done": ["closure", "quiet"]}
        )
        result = self._result(world)
        assert {r["id"] for r in result["entries"]} == {"closure", "quiet"}
        assert sorted(result["not_processed"]) == ["elsewhere", "storm"]
        assert result["summary"]["event_recall"]["n"] == 1

    def test_an_unfinished_run_without_progress_scores_nothing(self, world):
        run_id, _ = world
        self._set_run(run_id, status="failed")
        result = self._result(world)
        assert result["entries"] == []
        assert len(result["not_processed"]) == 4

    def test_no_lead_time_without_its_disruption(self, world):
        run_id, _ = world
        self._set_run(run_id, manifest={"calendar_done": ["storm"]})
        assert self._result(world)["lead_times"] == []

    def test_unique_counts_what_survives_deduplication(self, world):
        rows = {r["id"]: r for r in self._result(world)["entries"]}
        assert rows["closure"]["unique"] == 2

    def test_cost_is_counted_per_window_and_per_run(self, world):
        run_id, _ = world
        with session_scope() as session:
            for verdict in session.scalars(select(Verdict).where(Verdict.run_id == run_id)):
                verdict.input_tokens, verdict.output_tokens, verdict.latency_s = 100, 10, 2.0
        result = self._result(world)
        rows = {r["id"]: r for r in result["entries"]}
        # hit and seen are the Hong Kong window's documents: three verdicts.
        assert rows["closure"]["cost"] == {
            "pairs": 3,
            "input_tokens": 300,
            "output_tokens": 30,
            "seconds": 6.0,
        }
        assert result["cost"]["pairs"] == 4

    def _review(self, entry_id: str, url: str, confirmed: bool) -> None:
        with session_scope() as session:
            session.add(
                CalendarReview(entry_id=entry_id, document_url=url, confirmed=confirmed)
            )

    def test_an_unreviewed_detection_is_pending(self, world):
        result = self._result(world)
        rows = {r["id"]: r for r in result["entries"]}
        assert rows["closure"]["verified"] is None
        assert "closure" in result["summary"]["verified"]["pending_review"]
        assert result["summary"]["verified"]["event_recall"]["hits"] == 0

    def test_a_confirmed_match_verifies_the_detection(self, world):
        self._review("closure", "https://cal.test/hit", True)
        result = self._result(world)
        rows = {r["id"]: r for r in result["entries"]}
        assert rows["closure"]["verified"] is True
        assert result["summary"]["verified"]["event_recall"]["hits"] == 1

    def test_a_rejected_match_is_a_coincidence_not_a_detection(self, world):
        """Matched, but the article was about another event of the same kind."""
        self._review("closure", "https://cal.test/hit", False)
        result = self._result(world)
        rows = {r["id"]: r for r in result["entries"]}
        assert rows["closure"]["detected"] is True
        assert rows["closure"]["verified"] is False
        assert result["summary"]["verified"]["event_recall"]["hits"] == 0

    def test_reviews_import_from_csv_and_update_in_place(self, world, tmp_path):
        from typer.testing import CliRunner

        from hontology.cli import app

        path = tmp_path / "review.csv"
        path.write_text(
            "entry_id,url,confirmed,note\n"
            "closure,https://cal.test/hit,no,first look\n"
            "closure,https://cal.test/seen,,\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        assert runner.invoke(app, ["eval", "calendar-review-import", str(path)]).exit_code == 0
        path.write_text(
            "entry_id,url,confirmed,note\nclosure,https://cal.test/hit,yes,second look\n",
            encoding="utf-8",
        )
        assert runner.invoke(app, ["eval", "calendar-review-import", str(path)]).exit_code == 0
        with session_scope() as session:
            reviews = list(session.scalars(select(CalendarReview)))
        assert [(r.confirmed, r.note) for r in reviews] == [(True, "second look")]
