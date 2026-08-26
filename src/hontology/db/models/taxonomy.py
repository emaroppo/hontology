"""External code systems and their links to concepts.

The reference schema kept one reference table and one association table per code
level (root / base / event), which meant three near-identical model pairs and a
level→table dispatch map threaded through every caller. Collapsed here into one
`Code` table carrying a ``level`` and a ``parent_code_id``, and one `ConceptCode`
association. Same semantics, a third of the surface, and it generalizes to a code
system that isn't three levels deep.

The invariant worth preserving exactly is on `ConceptCode.similarity_run_id`:

    a link proposed automatically carries the run id that proposed it;
    a link a human added by hand has NULL.

Recomputing similarity deletes and rebuilds only the rows with a run id. That is
the whole reason curation survives a recompute, and it is covered by a test.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, TimestampMixin


class CodeSystem(Base, TimestampMixin):
    """A named external taxonomy (e.g. the CAMEO event codebook)."""

    __tablename__ = "code_systems"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    version: Mapped[str | None] = mapped_column(String)


class Code(Base, TimestampMixin):
    """One code in a system.

    ``level`` orders granularity from coarse to fine (for CAMEO: ``root`` 2-digit,
    ``base`` 3-digit, ``event`` 4-digit). ``parent_code_id`` makes the fallback
    chain walkable without string slicing.
    """

    __tablename__ = "codes"
    __table_args__ = (UniqueConstraint("system_id", "code"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    system_id: Mapped[int] = mapped_column(
        ForeignKey("code_systems.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(Text)
    level: Mapped[str] = mapped_column(String, nullable=False, index=True)
    parent_code_id: Mapped[int | None] = mapped_column(ForeignKey("codes.id"))

    system: Mapped[CodeSystem] = relationship()
    parent: Mapped[Code | None] = relationship(remote_side="Code.id")


class ConceptCode(Base, TimestampMixin):
    """Curated concept↔code association.

    ``similarity_run_id`` NULL means a human added this by hand and a recompute
    must leave it alone.
    """

    __tablename__ = "concept_codes"
    __table_args__ = (UniqueConstraint("concept_id", "code_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    concept_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code_id: Mapped[int] = mapped_column(
        ForeignKey("codes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity_score: Mapped[float | None] = mapped_column(Float)
    similarity_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("similarity_runs.id", ondelete="SET NULL"), index=True
    )

    code: Mapped[Code] = relationship()
