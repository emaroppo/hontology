"""Near-duplicate grouping, and copies reading their representative's verdicts.

The stability tests matter most: a representative carries a group's verdicts,
so re-running over a wider set must never move it or chain one copy to another.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from hontology.db.models import Document
from hontology.db.session import session_scope
from hontology.ingest import dedup

STORY = (
    "Typhoon Ragasa forced the suspension of container terminal operations in Hong Kong "
    "on Wednesday as the observatory raised its highest storm signal, and shipping lines "
    "warned customers that vessel calls across the Pearl River Delta would be delayed "
    "until the storm passed and port authorities had inspected berths and cranes for "
    "damage, with some services expected to resume on Thursday afternoon at the earliest."
)
OTHER = (
    "Dockworkers in Rotterdam walked out on Wednesday in a dispute over lashing rules, "
    "stopping work at several container terminals for the second time this month and "
    "leaving ships waiting at anchor while the union and employers traded proposals "
    "through a mediator appointed by the port authority to end the long running dispute."
)


class TestSignatures:
    def test_a_lightly_edited_copy_scores_high(self):
        edited = STORY.replace("Wednesday", "Wednesday morning") + " Reporting by staff."
        score = dedup.similarity(
            dedup.signature(dedup.shingles(STORY)), dedup.signature(dedup.shingles(edited))
        )
        assert score >= 0.8

    def test_different_stories_score_low(self):
        score = dedup.similarity(
            dedup.signature(dedup.shingles(STORY)), dedup.signature(dedup.shingles(OTHER))
        )
        assert score < 0.2

    def test_clusters_group_copies_and_leave_singletons_out(self):
        sigs = {
            1: dedup.signature(dedup.shingles(STORY)),
            2: dedup.signature(dedup.shingles(STORY + " Copyright wire service.")),
            3: dedup.signature(dedup.shingles(OTHER)),
        }
        assert dedup.clusters(sigs) == [{1, 2}]


@pytest.mark.requires_db
class TestDeduplicate:
    @pytest.fixture
    def make(self, tmp_path, monkeypatch):
        from hontology.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))

        def make_document(name: str, text: str, *, duplicate_of: int | None = None) -> int:
            (tmp_path / f"{name}.txt").write_text(text)
            with session_scope() as session:
                doc = Document(
                    url=f"https://dup.test/{name}",
                    url_hash=f"dup{name:>013}",
                    body_path=f"{name}.txt",
                    duplicate_of=duplicate_of,
                )
                session.add(doc)
                session.flush()
                return doc.id

        return make_document

    @staticmethod
    def _marks(ids: list[int]) -> dict[int, int | None]:
        with session_scope() as session:
            return dict(
                session.execute(
                    select(Document.id, Document.duplicate_of).where(Document.id.in_(ids))
                ).all()
            )

    def test_the_earliest_copy_becomes_the_representative(self, make):
        first = make("a", STORY)
        second = make("b", STORY + " Additional reporting.")
        other = make("c", OTHER)
        with session_scope() as session:
            result = dedup.deduplicate(session, [first, second, other])
        assert self._marks([first, second, other]) == {first: None, second: first, other: None}
        assert result["representatives"] == 2

    def test_an_existing_representative_is_never_moved(self, make):
        """An older, never-grouped copy joins the group; it must not take over,
        or the verdicts already on the representative would be orphaned."""
        older = make("x", STORY)
        representative = make("y", STORY)
        copy = make("z", STORY, duplicate_of=representative)
        with session_scope() as session:
            dedup.deduplicate(session, [older, copy])
        marks = self._marks([older, representative, copy])
        assert marks == {older: representative, representative: None, copy: representative}

    def test_short_texts_stay_unique(self, make):
        a = make("s1", "Port closed.")
        b = make("s2", "Port closed.")
        with session_scope() as session:
            result = dedup.deduplicate(session, [a, b])
        assert result["too_short"] == 2
        assert self._marks([a, b]) == {a: None, b: None}
