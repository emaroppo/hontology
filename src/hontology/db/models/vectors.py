"""Embeddings and similarity provenance.

A similarity result is only trustworthy if you can say afterwards *which* model
compared *which* text against *which* target. `SimilarityRun` records that spec
and every pairwise score is persisted, so a proposed link stays auditable long
after the run that produced it.

`Embedding.text_key` is content-addressed: ``"<fields>@<hash-of-text>"``. Keying
on the field composition alone (``"name"``, ``"name+definition"``) would let an
edited concept keep its stale vector under the same key, and similarity would
silently go on scoring the old text. Hashing the exact embedded text into the key
makes a changed definition miss the cache and get re-embedded, and lets old
versions coexist for replay.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, TimestampMixin


class EmbeddingModel(Base, TimestampMixin):
    __tablename__ = "embedding_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    provider: Mapped[str | None] = mapped_column(String)
    model_name: Mapped[str | None] = mapped_column(String)
    dim: Mapped[int | None] = mapped_column(Integer)
    normalized: Mapped[bool] = mapped_column(default=True)


class Embedding(Base, TimestampMixin):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("model_id", "object_type", "object_id", "text_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # "concept" | "code" | "document"
    object_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    object_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # Content-addressed: "<fields>@<sha256[:12] of source_text>". See module docstring.
    text_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_text: Mapped[str | None] = mapped_column(Text)
    # Dimension is left unspecified: every row for a given model shares that
    # model's dim, so cosine operations within a model are well-defined.
    embedding: Mapped[list[float]] = mapped_column(Vector())


class SimilarityRun(Base, TimestampMixin):
    """The full spec of one similarity computation."""

    __tablename__ = "similarity_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("embedding_models.id"), nullable=False)
    source_object_type: Mapped[str] = mapped_column(String, nullable=False)
    source_text_key: Mapped[str] = mapped_column(String, nullable=False)
    target_object_type: Mapped[str] = mapped_column(String, nullable=False)
    target_text_key: Mapped[str] = mapped_column(String, nullable=False)
    metric: Mapped[str] = mapped_column(String, default="cosine", nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    model: Mapped[EmbeddingModel] = relationship()


class SimilarityScore(Base):
    """Every pairwise score, not just the ones above threshold.

    Keeping the sub-threshold scores is what lets the review UI show near-misses
    and lets a threshold be re-chosen after the fact without recomputing.
    """

    __tablename__ = "similarity_scores"
    __table_args__ = (UniqueConstraint("run_id", "source_id", "target_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("similarity_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
