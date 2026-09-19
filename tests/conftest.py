"""Test isolation.

Two layers, because one is not enough:

**A separate database.** Tests never touch the development database. The URL is
derived by suffixing the configured database name with ``_test`` (override with
``HONTOLOGY_TEST_DATABASE_URL``), it is created and migrated on first use, and a
guard refuses to run if the resulting name does not end in ``_test``. Without
this a stray unqualified ``DELETE`` in a test destroys real work — which is
exactly how these fixtures came to exist.

**A transaction per test.** Each test runs inside a transaction that is rolled
back afterwards, so tests cannot see or clobber each other's rows either. The
session factory is bound to the test's connection with
``join_transaction_mode="create_savepoint"``, which turns the ``commit()`` calls
inside application code into savepoint releases rather than real commits. That
matters here because the judge loop commits per pair on purpose, and a naive
rollback fixture would have nothing left to roll back.

The combination means a test can be as destructive as it likes and still leave
nothing behind.

**Missing services skip, they do not error.** Postgres and the model provider are
each probed once per session. When one is unreachable, the tests marked
``requires_db`` or ``requires_llm`` are skipped with a reason naming the command
that starts it, so a fresh clone can run everything that does not need them.
Settings still point at the ``_test`` database either way, so an unmarked test
that reaches for Postgres fails loudly rather than finding the development data.
Set ``HONTOLOGY_TEST_REQUIRE_SERVICES=1`` to turn those skips back into failures,
so a CI job cannot pass by skipping everything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from hontology import config as config_module
from hontology.db import session as session_module


def _test_database_url() -> str:
    explicit = os.environ.get("HONTOLOGY_TEST_DATABASE_URL")
    if explicit:
        return explicit

    base = config_module.Settings().database_url
    parts = urlsplit(base)
    name = parts.path.lstrip("/")
    return urlunsplit(parts._replace(path=f"/{name}_test"))


def _admin_url(url: str) -> str:
    """The same server, pointed at the default database, for CREATE DATABASE."""
    parts = urlsplit(url.replace("postgresql+psycopg://", "postgresql://"))
    return urlunsplit(parts._replace(path="/postgres"))


def _database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


def _ensure_database(url: str) -> None:
    name = _database_name(url)
    with psycopg.connect(_admin_url(url), autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            # Identifier cannot be parameterised; the name is derived from our own
            # settings and validated by the _test suffix guard below.
            connection.execute(f'CREATE DATABASE "{name}"')


def _postgres_reachable(url: str) -> bool:
    try:
        with psycopg.connect(_admin_url(url), connect_timeout=3):
            return True
    except psycopg.Error:
        return False


def _llm_reachable() -> bool:
    host = config_module.Settings().ollama_host.rstrip("/")
    try:
        return httpx.get(f"{host}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


_SERVICES = {
    "requires_db": (
        lambda: _postgres_reachable(_test_database_url()),
        "Postgres is not reachable (start it with `make up`)",
    ),
    "requires_llm": (_llm_reachable, "the model provider is not reachable (`ollama serve`)"),
}

# Whether this session builds and uses the test database: only when a selected
# test needs it and Postgres answered.
_DB_READY = pytest.StashKey[bool]()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    strict = os.environ.get("HONTOLOGY_TEST_REQUIRE_SERVICES") == "1"
    for marker, (probe, reason) in _SERVICES.items():
        wanted = [item for item in items if item.get_closest_marker(marker)]
        # Probe only when something selected needs the service: `-m "not
        # requires_db"` should not pay for a connection timeout.
        up = bool(wanted) and (strict or probe())
        if marker == "requires_db":
            config.stash[_DB_READY] = up
        if wanted and not up:
            for item in wanted:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session", autouse=True)
def test_database(pytestconfig: pytest.Config) -> Iterator[str]:
    """Point the whole application at a dedicated, migrated test database."""
    url = _test_database_url()
    name = _database_name(url)

    # The guard that makes the rest of this safe.
    if not name.endswith("_test"):
        raise RuntimeError(
            f"refusing to run tests against database {name!r}: the name must end "
            "in '_test'. Set HONTOLOGY_TEST_DATABASE_URL if you need a different one."
        )

    # Every layer reads settings through get_settings(), and the engine and
    # session factory are cached, so all three have to be reset together. This
    # happens even when Postgres is down, so nothing can fall back to the
    # development database.
    original = config_module.get_settings
    settings = config_module.Settings(database_url=url)
    config_module.get_settings = lambda: settings  # type: ignore[assignment]
    session_module.get_engine.cache_clear()
    session_module.get_session_factory.cache_clear()

    if not pytestconfig.stash.get(_DB_READY, False):
        yield url
        config_module.get_settings = original  # type: ignore[assignment]
        return

    _ensure_database(url)

    engine = create_engine(url, pool_pre_ping=True, future=True)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    from hontology.db.models import Base

    Base.metadata.create_all(engine)

    # Country codes are reference data the application seeds at startup, not user
    # content, so tests need them present for anything that resolves a locus.
    # Seeded once outside the per-test transaction so every test sees them.
    from hontology.ingest.loci import seed_loci

    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        seed_loci(session)
        session.commit()

    yield url

    engine.dispose()
    config_module.get_settings = original  # type: ignore[assignment]
    session_module.get_engine.cache_clear()
    session_module.get_session_factory.cache_clear()


@pytest.fixture(autouse=True)
def rollback_after_each_test(test_database: str, pytestconfig: pytest.Config) -> Iterator[None]:
    """Run each test inside a transaction and roll it back afterwards.

    ``join_transaction_mode="create_savepoint"`` is what makes this work against
    application code that commits: those commits become savepoint releases inside
    the outer transaction, which is then discarded whole.
    """
    if not pytestconfig.stash.get(_DB_READY, False):
        yield
        return
    engine = create_engine(test_database, pool_pre_ping=True, future=True)
    connection = engine.connect()
    transaction = connection.begin()

    factory = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        future=True,
        join_transaction_mode="create_savepoint",
    )

    original_engine = session_module.get_engine
    original_factory = session_module.get_session_factory
    session_module.get_engine = lambda: engine  # type: ignore[assignment]
    session_module.get_session_factory = lambda: factory  # type: ignore[assignment]

    try:
        yield
    finally:
        session_module.get_engine = original_engine  # type: ignore[assignment]
        session_module.get_session_factory = original_factory  # type: ignore[assignment]
        transaction.rollback()
        connection.close()
        engine.dispose()
