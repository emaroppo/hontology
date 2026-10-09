"""The pre-scrape code filter.

This finally exercises the tier-fallback rule, which lived in the old code
retrieval source and was never covered by a test there.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from hontology.db.models import Code, Document, FeedArticle, FeedEvent, FeedSlice
from hontology.db.session import session_scope
from hontology.ontology import service
from hontology.pipeline.ingest.articles import filter as ingest_filter
from hontology.pipeline.ingest.codes import cameo, themes
from hontology.pipeline.retrieve import similarity

pytestmark = pytest.mark.requires_db

SAMPLE = (
    "14\tPROTEST\n"
    "145\tProtest violently, riot\n"
    "1451\tEngage in political dissent, riot\n"
    "06\tENGAGE IN MATERIAL COOPERATION\n"
)


@pytest.fixture
def setup():
    """An ontology with one linked concept, plus a slice of feed events."""
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-filter", name="Filter")
        riot = service.create_concept(session, ontology.id, name="Riot", definition="A riot.")
        unlinked = service.create_concept(
            session, ontology.id, name="Flood", definition="A flood."
        )
        result = cameo.load_codes(session, cameo.parse_lookup(SAMPLE))
        codes = {
            c.code: c.id
            for c in session.scalars(select(Code).where(Code.system_id == result["system_id"]))
        }

        # Riot is linked to the *base* code 145 only.
        similarity.set_link(session, riot.id, codes["145"], linked=True)

        feed_slice = FeedSlice(
            feed="test_filter",
            slice_key="20260827000000",
            sliced_at=datetime.now(UTC),
            status="ok",
        )
        session.add(feed_slice)
        session.flush()

        ids = {
            "ontology": ontology.id,
            "riot": riot.id,
            "unlinked": unlinked.id,
            "codes": codes,
            "slice": feed_slice.id,
        }
    yield ids


def add_document(session, suffix: str) -> int:
    document = Document(url=f"https://flt.test/{suffix}", url_hash=f"flthash{suffix:>08}")
    session.add(document)
    session.flush()
    return document.id


def add_event(session, ids, document_id: int, *, event=None, base=None, root=None) -> None:
    session.add(
        FeedEvent(
            slice_id=ids["slice"],
            feed_event_id=f"e{document_id}-{event}-{base}-{root}",
            document_id=document_id,
            event_code=event,
            base_code=base,
            root_code=root,
        )
    )
    session.flush()


class TestTierFallback:
    def test_a_matching_base_code_admits_the_document(self, setup):
        with session_scope() as session:
            doc = add_document(session, "1")
            add_event(session, setup, doc, base="145", root="14")

        with session_scope() as session:
            matches = ingest_filter.matching_documents(session, setup["ontology"])
            assert doc in matches
            assert matches[doc].code == "145"
            assert matches[doc].level == "base"
            assert setup["riot"] in matches[doc].concept_ids

    def test_the_first_tier_carrying_a_code_decides(self, setup):
        """An event coded 1451 that matches nothing must NOT be retried as 145.

        Falling through would let a specific unmatched code match everything
        protest-shaped at a broader tier, which is far more than the event
        actually supports.
        """
        with session_scope() as session:
            doc = add_document(session, "2")
            # 1451 is a real code but is not linked; 145 IS linked.
            add_event(session, setup, doc, event="1451", base="145", root="14")

        with session_scope() as session:
            assert ingest_filter.matching_documents(session, setup["ontology"]) == {}

    def test_a_null_finer_tier_is_skipped_not_treated_as_a_decision(self, setup):
        """GDELT leaves the event tier empty when no finer code applies."""
        with session_scope() as session:
            doc = add_document(session, "3")
            add_event(session, setup, doc, event=None, base="145", root="14")

        with session_scope() as session:
            assert doc in ingest_filter.matching_documents(session, setup["ontology"])

    def test_an_unrelated_code_is_excluded(self, setup):
        with session_scope() as session:
            doc = add_document(session, "4")
            add_event(session, setup, doc, base="06")

        with session_scope() as session:
            assert ingest_filter.matching_documents(session, setup["ontology"]) == {}

    def test_an_event_with_no_codes_at_all_is_excluded(self, setup):
        with session_scope() as session:
            doc = add_document(session, "5")
            add_event(session, setup, doc)

        with session_scope() as session:
            assert ingest_filter.matching_documents(session, setup["ontology"]) == {}


class TestNoLinks:
    def test_an_unlinked_ontology_cannot_filter(self, setup):
        """Empty must read as 'cannot filter', never as 'nothing matches'."""
        with session_scope() as session:
            other = service.create_ontology(session, slug="test-filter-2", name="No links")
            service.create_concept(session, other.id, name="Riot", definition="A riot.")
            other_id = other.id

        with session_scope() as session:
            assert ingest_filter.concept_code_map(session, other_id) == {}
            report = ingest_filter.preview(session, other_id)
            assert report["usable"] is False
            assert "no concept" in report["reason"]

    def test_preview_reports_what_would_be_kept(self, setup):
        with session_scope() as session:
            keep = add_document(session, "6")
            add_event(session, setup, keep, base="145")
            drop = add_document(session, "7")
            add_event(session, setup, drop, base="06")

        with session_scope() as session:
            report = ingest_filter.preview(session, setup["ontology"])
            assert report["usable"] is True
            assert report["documents_matching"] >= 1
            assert report["unfetched_matching"] >= 1
            assert report["unfetched_skipped"] >= 1


class TestScrapeIntegration:
    def test_the_filter_gates_which_documents_are_fetched(self, setup, monkeypatch):
        """The whole point: the budget goes to documents the ontology can use."""
        from hontology.pipeline.ingest.articles import scrape

        with session_scope() as session:
            keep = add_document(session, "8")
            add_event(session, setup, keep, base="145")
            drop = add_document(session, "9")
            add_event(session, setup, drop, base="06")

        attempted: list[str] = []

        def fake_fetch(url, **kwargs):
            attempted.append(url)
            return None, 404, "HTTP 404"

        monkeypatch.setattr(scrape, "fetch_html", fake_fetch)
        monkeypatch.setattr(scrape.RobotsCache, "allowed", lambda self, url: True)

        with session_scope() as session:
            scrape.scrape_pending(session, limit=10, ontology_id=setup["ontology"])

        assert any("flt.test/8" in url for url in attempted)
        assert not any("flt.test/9" in url for url in attempted)

    def test_without_an_ontology_nothing_is_filtered(self, setup, monkeypatch):
        from hontology.pipeline.ingest.articles import scrape

        with session_scope() as session:
            doc = add_document(session, "10")
            add_event(session, setup, doc, base="06")

        attempted: list[str] = []
        monkeypatch.setattr(
            scrape, "fetch_html", lambda url, **kw: (attempted.append(url), (None, 404, "x"))[1]
        )
        monkeypatch.setattr(scrape.RobotsCache, "allowed", lambda self, url: True)

        with session_scope() as session:
            scrape.scrape_pending(session, limit=10)

        assert any("flt.test/10" in url for url in attempted)

    def test_an_unfilterable_ontology_fetches_nothing_and_says_so(self, setup, monkeypatch):
        """Silently fetching zero would look like the feed had gone quiet."""
        from hontology.pipeline.ingest.articles import scrape

        with session_scope() as session:
            other = service.create_ontology(session, slug="test-filter-3", name="No links")
            other_id = other.id
            doc = add_document(session, "11")
            add_event(session, setup, doc, base="145")

        monkeypatch.setattr(scrape.RobotsCache, "allowed", lambda self, url: True)

        with session_scope() as session:
            result = scrape.scrape_pending(session, limit=10, ontology_id=other_id)

        assert result["attempted"] == 0
        assert "note" in result["filter"]


class TestThemes:
    """The knowledge-graph half of the filter: articles with no coded event."""

    @pytest.fixture
    def themed(self, setup):
        with session_scope() as session:
            result = themes.load_themes(
                session,
                [("CYBER_ATTACK", 10), ("SHORTAGE", 5), ("SPORTS", 99)],
                source_url="test",
            )
            codes = {
                c.code: c.id
                for c in session.scalars(
                    select(Code).where(Code.system_id == result["system_id"])
                )
            }
            similarity.set_link(session, setup["unlinked"], codes["SHORTAGE"], linked=True)
        return setup

    def add_article(self, session, ids, document_id: int, article_themes: list[str]) -> None:
        session.add(
            FeedArticle(
                slice_id=ids["slice"],
                record_id=f"g{document_id}",
                document_id=document_id,
                themes=article_themes,
                locus_ids=[],
            )
        )
        session.flush()

    def test_a_linked_theme_admits_a_document_cameo_never_saw(self, themed):
        with session_scope() as session:
            doc = add_document(session, "t1")
            self.add_article(session, themed, doc, ["SHORTAGE", "SPORTS"])

        with session_scope() as session:
            matches = ingest_filter.matching_documents(session, themed["ontology"])
            assert matches[doc].code == "SHORTAGE"
            assert matches[doc].level == "theme"
            assert matches[doc].concept_ids == {themed["unlinked"]}

    def test_unlinked_themes_admit_nothing(self, themed):
        with session_scope() as session:
            doc = add_document(session, "t2")
            self.add_article(session, themed, doc, ["SPORTS", "CYBER_ATTACK"])

        with session_scope() as session:
            assert doc not in ingest_filter.matching_documents(session, themed["ontology"])

    def test_a_cameo_match_is_reported_first(self, themed):
        with session_scope() as session:
            doc = add_document(session, "t3")
            add_event(session, themed, doc, base="145", root="14")
            self.add_article(session, themed, doc, ["SHORTAGE"])

        with session_scope() as session:
            assert (
                ingest_filter.matching_documents(session, themed["ontology"])[doc].code == "145"
            )

    def test_theme_links_do_not_leak_into_the_cameo_map(self, themed):
        with session_scope() as session:
            assert "SHORTAGE" not in ingest_filter.concept_code_map(session, themed["ontology"])
            assert "SHORTAGE" in ingest_filter.concept_theme_map(session, themed["ontology"])


class TestLinkImport:
    def test_links_import_by_name_and_count_as_hand_made(self, setup):
        with session_scope() as session:
            result = similarity.import_links(
                session,
                setup["ontology"],
                [{"concept": "Flood", "system": "cameo", "code": "1451"}],
            )
            assert result == {"links": 1, "concepts": 1}
            links = ingest_filter.concept_code_map(session, setup["ontology"])
            assert links["1451"] == {setup["unlinked"]}

    def test_one_bad_row_applies_nothing(self, setup):
        with session_scope() as session, pytest.raises(LookupError, match="9999"):
            similarity.import_links(
                session,
                setup["ontology"],
                [
                    {"concept": "Flood", "system": "cameo", "code": "1451"},
                    {"concept": "Flood", "system": "cameo", "code": "9999"},
                ],
            )
        with session_scope() as session:
            assert "1451" not in ingest_filter.concept_code_map(session, setup["ontology"])
