"""A connection failure leaves the document pending, up to a limit.

A network blip and a dead host fail the same way, so neither is final on the
first try; only repeated failures in separate batches are recorded.
"""

from __future__ import annotations

import pytest

from hontology.db.models import Document
from hontology.db.session import session_scope
from hontology.pipeline.ingest.articles import scrape

pytestmark = pytest.mark.requires_db

DNS = "ConnectError: [Errno -2] Name or service not known"


@pytest.fixture
def document_id(monkeypatch, tmp_path) -> int:
    from hontology.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(scrape.RobotsCache, "allowed", lambda self, url: True)
    monkeypatch.setattr(scrape.RobotsCache, "crawl_delay", lambda self, url: None)
    with session_scope() as session:
        document = Document(url="https://flaky.test/a", url_hash="flakytest0001")
        session.add(document)
        session.flush()
        return document.id


def attempt(monkeypatch, document_id: int, error: str) -> tuple[dict, Document]:
    monkeypatch.setattr(scrape, "fetch_html", lambda url, **kw: (None, None, error))
    with session_scope() as session:
        result = scrape.scrape_pending(session, document_ids=[document_id])
    with session_scope() as session:
        return result, session.get(Document, document_id)


def test_a_connection_failure_leaves_the_document_pending(monkeypatch, document_id):
    result, document = attempt(monkeypatch, document_id, DNS)
    assert result["retry_later"] == 1
    assert result["attempted"] == 0  # so a batch of only these ends the loop
    assert document.fetched_at is None
    assert document.connect_failures == 1


def test_it_fails_for_good_at_the_limit(monkeypatch, document_id):
    for _ in range(scrape.MAX_CONNECT_FAILURES - 1):
        attempt(monkeypatch, document_id, DNS)
    result, document = attempt(monkeypatch, document_id, DNS)
    assert result["failed"] == 1
    assert document.fetched_at is not None
    assert document.error == DNS


def test_other_failures_are_final_at_once(monkeypatch, document_id):
    result, document = attempt(monkeypatch, document_id, "HTTP 404")
    assert result["failed"] == 1
    assert document.fetched_at is not None
    assert document.connect_failures == 0


def test_a_connection_failure_is_handed_back_and_can_be_left_out(monkeypatch, document_id):
    result, _ = attempt(monkeypatch, document_id, DNS)
    assert result["held_back"] == [document_id]
    with session_scope() as session:
        assert scrape.pending_documents(session, 10, document_ids=[document_id]) != []
        assert (
            scrape.pending_documents(
                session, 10, document_ids=[document_id], exclude=[document_id]
            )
            == []
        )
