"""Stored retrievals: what each article was shown to the judge, and how.

Retrieval is a sub-pipeline with fixed ends (an article in, the leaves each
deciding call is shown out) and steps that vary by strategy; see
`pipeline.retrieve.strategies`. A run stores each article's retrieval when it
first retrieves it, before any deciding call:

- `ArticleRetrieval` is the article's retrieval as a whole: the strategy, its
  trace, the routing answers and events later steps produced, why it stopped
  early if it did, and what its model calls cost.
- `ShownLeaf` is one leaf in one deciding call. These rows are the shown
  record: a pair is *retrieved* if and only if it has one.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from hontology.db.base import Base, TimestampMixin


class ArticleRetrieval(Base, TimestampMixin):
    __tablename__ = "article_retrievals"
    __table_args__ = (UniqueConstraint("run_id", "document_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # "embedding" | "routing" | "extraction" | "extraction, ranked"
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    # One entry per step taken, as the step noted it.
    trace: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Routing calls over parent classes: classes asked, parsed replies, cost.
    asked: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Events found, with the parents each was routed through and leaves reached.
    events: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Parent class id -> answered yes for any event (event strategies).
    parent_yes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    stopped: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # The run that computed it, when this run adopted it rather than computing it.
    copied_from_run: Mapped[int | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL")
    )


class ShownLeaf(Base):
    """One leaf in one deciding call for one article."""

    __tablename__ = "shown_leaves"
    __table_args__ = (UniqueConstraint("run_id", "document_id", "offer", "concept_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Which deciding call, in order, for this article.
    offer: Mapped[int] = mapped_column(Integer, nullable=False)
    # For event strategies: the ordinal of the event the call was about.
    event: Mapped[int | None] = mapped_column(Integer)
