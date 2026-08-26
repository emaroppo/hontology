"""Alembic environment.

The database URL comes from application settings rather than alembic.ini, so
there is exactly one place a connection string is configured.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool, text

from hontology.config import get_settings
from hontology.db.models import Base  # imports every model


def render_item(type_, obj, autogen_context) -> str | bool:
    """Teach autogenerate to emit pgvector columns with their import.

    Without this, a generated migration references ``pgvector.sqlalchemy`` but
    never imports it, and the migration fails with a NameError the first time it
    is applied on a clean database.
    """
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        dim = getattr(obj, "dim", None)
        return f"pgvector.sqlalchemy.Vector({dim if dim else ''})"
    return False

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # pgvector must exist before any Vector column is created.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
