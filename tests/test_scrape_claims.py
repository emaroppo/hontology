"""Scrapers running side by side claim disjoint documents.

Claims are row locks, which only another connection can see, so these tests
commit on connections of their own instead of the per-test transaction, and
delete what they made.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, sessionmaker

from hontology.db.models import Document
from hontology.ingest.scrape import pending_documents, wait_for_claimed

pytestmark = pytest.mark.requires_db


@pytest.fixture
def two_sessions(test_database: str) -> Iterator[tuple[Session, Session, list[int]]]:
    engine = create_engine(test_database, future=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as setup:
        documents = [
            Document(url=f"https://claims.test/{i}", url_hash=f"claimstest{i:04d}")
            for i in range(4)
        ]
        setup.add_all(documents)
        setup.commit()
        ids = [d.id for d in documents]
    first, second = factory(), factory()
    try:
        yield first, second, ids
    finally:
        first.rollback()
        second.rollback()
        first.close()
        second.close()
        with factory() as cleanup:
            cleanup.execute(delete(Document).where(Document.id.in_(ids)))
            cleanup.commit()
        engine.dispose()


def test_a_claimed_document_is_skipped_by_the_other_scraper(two_sessions):
    first, second, ids = two_sessions
    mine = {d.id for d in pending_documents(first, 2, document_ids=ids, claim=True)}
    theirs = {d.id for d in pending_documents(second, 10, document_ids=ids, claim=True)}
    assert len(mine) == 2
    assert theirs == set(ids) - mine


def test_without_a_claim_nothing_is_locked(two_sessions):
    first, second, ids = two_sessions
    pending_documents(first, 10, document_ids=ids)
    assert len(pending_documents(second, 10, document_ids=ids, claim=True)) == 4


def test_waiting_returns_once_the_claim_is_committed(two_sessions):
    """The window's scraper waits for the other to finish its batch, so no
    document is retrieved before it has been fetched."""
    first, second, ids = two_sessions
    claimed = pending_documents(first, 4, document_ids=ids, claim=True)
    waited: list[bool] = []
    waiter = threading.Thread(target=lambda: waited.append(wait_for_claimed(second, ids)))
    waiter.start()
    waiter.join(timeout=1)
    assert waiter.is_alive()  # blocked on the claim

    for document in claimed:
        document.fetched_at = datetime.now(UTC)
    first.commit()
    waiter.join(timeout=10)
    assert waited == [True]
    assert wait_for_claimed(second, ids) is False  # nothing held any more


def test_one_scraper_at_a_time_holds_a_host():
    from hontology.ingest.scrape import host_lock

    order: list[str] = []

    def other() -> None:
        with host_lock("claims.test"):
            order.append("other")

    with host_lock("claims.test"):
        waiter = threading.Thread(target=other)
        waiter.start()
        waiter.join(timeout=1)
        assert waiter.is_alive()  # waiting for the host
        order.append("first")
    waiter.join(timeout=10)
    assert order == ["first", "other"]
    with host_lock("another.test"):
        pass  # a different host is never held up


def test_more_ids_than_postgres_allows_parameters():
    """A calendar's windows hold more documents than 65,535, the most
    parameters one query may bind; the ids travel as a single array."""
    from hontology.db.session import session_scope

    ids = list(range(1, 70_001))
    with session_scope() as session:
        assert pending_documents(session, 10, document_ids=ids) is not None
