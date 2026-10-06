"""Declarative base and column conventions shared by every model."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, DateTime, ForeignKey, Integer, any_, bindparam, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def among(column: Any, ids: Iterable[int]) -> ColumnElement[bool]:
    """``column = ANY(:ids)``, the ids sent as one array parameter.

    ``IN (...)`` binds a parameter per id, and Postgres refuses a query with
    more than 65,535 of them: a calendar's windows hold more documents than that.
    """
    return column == any_(bindparam(None, list(ids), type_=ARRAY(Integer)))


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class OwnedMixin:
    """A nullable owner, carried from the first migration.

    The application is deliberately single-user and has no authentication. This
    column exists anyway so that adding tenancy later is a migration and a filter
    rather than a reshaping of every table. Until then it stays NULL everywhere
    and nothing reads it.
    """

    owner_id: Mapped[int | None] = mapped_column(
        ForeignKey("owners.id", ondelete="CASCADE"), default=None, index=True
    )
