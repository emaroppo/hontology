"""Declarative base and column conventions shared by every model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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
