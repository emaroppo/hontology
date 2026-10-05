"""The Global Knowledge Graph feed: parsing, and ingest scoped to places.

The scoped-ingest tests guard the one way a partial backfill could quietly lose
data: a slice kept for one country being treated as done when another country is
asked for later.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from sqlalchemy import select

from hontology.db.models import FeedArticle, FeedSlice
from hontology.db.session import session_scope
from hontology.ingest import gdelt, service
from hontology.ingest.loci import by_fips

KEY = "20250923120000"


def _record(
    record_id: str,
    url: str,
    *,
    themes: str = "",
    themes_v2: str = "",
    locations: str = "",
    collection: str = "1",
    quotations: str = "",
) -> str:
    row = [""] * 27
    row[gdelt.GKG_COL_RECORD_ID] = record_id
    row[gdelt.GKG_COL_DATE] = KEY
    row[gdelt.GKG_COL_COLLECTION] = collection
    row[gdelt.GKG_COL_DOCUMENT] = url
    row[gdelt.GKG_COL_THEMES] = themes
    row[gdelt.GKG_COL_THEMES_V2] = themes_v2
    row[gdelt.GKG_COL_LOCATIONS] = locations
    row[22] = quotations
    return "\t".join(row)


def _zip(*lines: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{KEY}.gkg.csv", "\n".join(lines) + "\n")
    return buffer.getvalue()


# Hong Kong (FIPS HK) and the United States (FIPS US).
HK = "1#Hong Kong#HK#HK00#22.25#114.167#-1"
US = "1#United States#US#US#39.828#-98.5795#US"

PAYLOAD = _zip(
    _record("r1", "https://a.test/port", themes="MARITIME;WB_167_PORTS;", locations=HK),
    _record("r2", "https://b.test/tariff", themes="ECON_TAXATION;", locations=US),
    _record("r3", "https://c.test/both", themes="STRIKE;", locations=f"{HK};{US}"),
)


class TestParse:
    def test_themes_and_countries_are_read(self):
        records = {r["record_id"]: r for r in gdelt.parse_gkg(PAYLOAD)}
        assert records["r1"]["themes"] == {"MARITIME", "WB_167_PORTS"}
        assert records["r3"]["countries"] == {"HK", "US"}
        assert records["r1"]["url"] == "https://a.test/port"

    def test_only_web_articles_are_kept(self):
        """Other collections identify documents by something other than a URL."""
        payload = _zip(_record("r1", "not-a-url", collection="2", locations=US))
        assert list(gdelt.parse_gkg(payload)) == []

    def test_v2_themes_are_the_fallback(self):
        payload = _zip(
            _record("r1", "https://a.test", themes_v2="CYBER_ATTACK,120;SHORTAGE,40")
        )
        assert next(gdelt.parse_gkg(payload))["themes"] == {"CYBER_ATTACK", "SHORTAGE"}

    def test_a_stray_quote_does_not_swallow_the_next_record(self):
        """A quoting-aware reader would merge r2 into r1's quotation field."""
        payload = _zip(
            _record("r1", "https://a.test", quotations='he said "the port is'),
            _record("r2", "https://b.test"),
        )
        assert [r["record_id"] for r in gdelt.parse_gkg(payload)] == ["r1", "r2"]


class TestCovers:
    def test_a_full_slice_covers_any_request(self):
        assert service._covers(None, [1, 2])
        assert service._covers(None, None)

    def test_a_scoped_slice_covers_only_its_own_places(self):
        assert service._covers([1, 2], [1])
        assert not service._covers([1], [1, 2])

    def test_a_scoped_slice_never_covers_a_full_request(self):
        assert not service._covers([1, 2], None)


@pytest.mark.requires_db
class TestScopedIngest:
    @pytest.fixture
    def fetched(self, monkeypatch):
        calls: list[str] = []

        def fake_fetch(base_url, key, *, kind, timeout=60.0):
            calls.append(kind)
            return PAYLOAD

        monkeypatch.setattr(gdelt, "fetch_slice", fake_fetch)
        return calls

    @staticmethod
    def _loci():
        with session_scope() as session:
            lookup = by_fips(session)
            return lookup["HK"].id, lookup["US"].id

    @staticmethod
    def _stored() -> tuple[set[str], list[int] | None]:
        with session_scope() as session:
            records = set(
                session.scalars(
                    select(FeedArticle.record_id)
                    .join(FeedSlice, FeedSlice.id == FeedArticle.slice_id)
                    .where(FeedSlice.feed == service.FEED_GKG, FeedSlice.slice_key == KEY)
                )
            )
            scope = session.scalar(
                select(FeedSlice.scope).where(
                    FeedSlice.feed == service.FEED_GKG, FeedSlice.slice_key == KEY
                )
            )
            return records, scope

    def _ingest(self, loci):
        with session_scope() as session:
            return service.ingest_slice(session, KEY, feed=service.FEED_GKG, loci=loci)

    def test_only_articles_mentioning_the_scope_are_kept(self, fetched):
        hk, _ = self._loci()
        result = self._ingest([hk])
        assert (result["rows"], result["kept"]) == (3, 2)
        assert self._stored() == ({"r1", "r3"}, [hk])
        assert fetched == ["gkg"]

    def test_a_covered_request_is_skipped(self, fetched):
        hk, _ = self._loci()
        self._ingest([hk])
        assert self._ingest([hk])["skipped"] is True
        assert len(fetched) == 1

    def test_a_new_place_reprocesses_and_widens_the_scope(self, fetched):
        """The case a plain status check gets wrong: 'ok' for Hong Kong is not
        'ok' for the United States."""
        hk, us = self._loci()
        self._ingest([hk])
        self._ingest([us])
        assert self._stored() == ({"r1", "r2", "r3"}, sorted([hk, us]))
        assert len(fetched) == 2

    def test_an_unscoped_request_makes_the_slice_full(self, fetched):
        hk, _ = self._loci()
        self._ingest([hk])
        self._ingest(None)
        assert self._stored() == ({"r1", "r2", "r3"}, None)

    def test_articles_link_to_documents_with_their_places(self, fetched):
        hk, us = self._loci()
        self._ingest(None)
        with session_scope() as session:
            article = session.scalar(select(FeedArticle).where(FeedArticle.record_id == "r3"))
            assert article.document is not None
            assert article.document.url == "https://c.test/both"
            assert sorted(article.locus_ids) == sorted([hk, us])
            assert article.themes == ["STRIKE"]
