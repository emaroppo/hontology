"""Change-triggered ontology snapshots.

A snapshot pins the *text* of an ontology's concepts at a point in time, so that:

- an old run can be replayed against the definitions it was actually scored on, and
- a ground-truth label can be stamped with the wording it was judged against.

Snapshots are minted **only when the content changes**. The hash covers
``(id, name, definition, inclusion, exclusion)`` across the whole concept set; if
that hash is already registered the existing ``vN`` is reused. Minting on every
save would produce a new version for every keystroke and make the stamp useless.

``weight`` is deliberately excluded from the hash: it is a routing and scoring
number that never reaches the embedding or the prompt, so changing it cannot
invalidate a label or fork a retrieval artifact.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, TimestampMixin
from hontology.db.models.ontology import Ontology


class OntologySnapshot(Base, TimestampMixin):
    __tablename__ = "ontology_snapshots"
    __table_args__ = (
        UniqueConstraint("ontology_id", "version"),
        UniqueConstraint("ontology_id", "content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Human-facing identity: "v1", "v2", ...
    version: Mapped[str] = mapped_column(String, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    n_concepts: Mapped[int] = mapped_column(Integer, default=0)
    # The full concept set as JSON, so a replay never needs the live tables.
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    # The subclass_of edges as JSON [[child id, parent id], ...]; NULL for a
    # flat ontology. Kept apart from `payload` so code reading the concept rows
    # is unaffected, and hashed only when present.
    edges: Mapped[str | None] = mapped_column(Text)

    ontology: Mapped[Ontology] = relationship()


class LinkSnapshot(Base, TimestampMixin):
    """The pre-download filter's code links at a point in time.

    Links are live rows a tick changes, so a run that applied the filter would
    otherwise leave no record of which links were in force. Minted only when the
    set changes, like ontology versions: "f1", "f2", ... per ontology.
    """

    __tablename__ = "link_snapshots"
    __table_args__ = (
        UniqueConstraint("ontology_id", "version"),
        UniqueConstraint("ontology_id", "content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    n_links: Mapped[int] = mapped_column(Integer, default=0)
    # [[concept id, system slug, code], ...], sorted.
    payload: Mapped[str] = mapped_column(Text, nullable=False)

    ontology: Mapped[Ontology] = relationship()
