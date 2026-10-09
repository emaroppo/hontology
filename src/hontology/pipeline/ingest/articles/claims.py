"""Claiming pending documents, so scrapers running side by side never collide.

Rows are claimed with ``SELECT ... FOR UPDATE SKIP LOCKED``, and a host is held
across every scraper on the database with an advisory lock, so two scrapers
neither fetch the same document nor double the rate one host sees.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import ColumnElement, Engine, create_engine, not_, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.lookups import count
from hontology.db.models import Document


def pending_documents(
    session: Session,
    limit: int,
    *,
    retry_failed: bool = False,
    document_ids: list[int] | None = None,
    claim: bool = False,
    exclude: list[int] | None = None,
) -> list[Document]:
    """Documents that still need a body.

    By default only never-attempted ones, so a run does not spend its budget
    re-hitting URLs already known to be dead. ``document_ids`` narrows the work
    to a specific set — useful for fetching exactly the documents one run needs
    rather than draining the whole backlog.

    With *claim*, the rows are locked until the caller commits, and rows another
    scraper has claimed are skipped, so scrapers running side by side never
    fetch the same document or double the rate on a host.

    *exclude* leaves out documents a caller is holding back for now, such as
    ones whose host asked for a long crawl delay.
    """
    query = select(Document)
    query = (
        query.where(Document.fetched_at.is_(None))
        if not retry_failed
        else query.where(Document.body_path.is_(None))
    )
    if document_ids is not None:
        query = query.where(among(Document.id, document_ids))
    if exclude:
        query = query.where(not_(among(Document.id, exclude)))
    query = query.order_by(Document.id).limit(limit)
    if claim:
        query = query.with_for_update(skip_locked=True, of=Document)
    return list(session.scalars(query))


@lru_cache
def _lock_engine(url: str) -> Engine:
    # Unpooled: each host holds its own connection only while it is scraped,
    # and a pool sized for the app would make the workers queue for one.
    return create_engine(url, poolclass=NullPool, future=True)


@contextmanager
def host_lock(host: str) -> Iterator[None]:
    """Hold a host for this process, across every scraper on the database.

    The per-host spacing is kept inside one process; a second scraper working
    on other documents from the same host would double the rate it sees. This
    waits until no other scraper is on the host.
    """
    with _lock_engine(get_settings().database_url).connect() as connection:
        key = {"host": host}
        connection.execute(text("SELECT pg_advisory_lock(hashtextextended(:host, 0))"), key)
        try:
            yield
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(hashtextextended(:host, 0))"), key
            )


def count_pending(
    session: Session, *, retry_failed: bool, document_ids: list[int] | None
) -> int:
    where: list[ColumnElement[bool]] = [
        Document.body_path.is_(None) if retry_failed else Document.fetched_at.is_(None)
    ]
    if document_ids is not None:
        where.append(among(Document.id, document_ids))
    return count(session, Document.id, *where)
