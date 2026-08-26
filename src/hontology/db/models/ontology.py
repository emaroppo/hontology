"""The user-defined ontology.

An `Ontology` is a namespace the user fills in through the UI; nothing about the
domain is baked into the schema. Three structural decisions are load-bearing:

- **Per-edge weighting** on `ConceptGroupMember` rather than only on `Concept`,
  so a forked group can retune the importance of a shared concept without
  touching the original or duplicating it.
- **Self-referential `parent_id`** on `ConceptGroup`, which is what makes those
  forks ("this group, but with these weights") cheap.
- **Inclusion and exclusion criteria as first-class fields** on `Concept`. They
  are not documentation: they go into the judge prompt verbatim, and separating
  them from the definition is what lets a user tighten a concept's boundary
  without rewriting what it means.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, OwnedMixin, TimestampMixin


class Owner(Base, TimestampMixin):
    """Placeholder tenancy anchor. Unused while the app is single-user."""

    __tablename__ = "owners"

    id: Mapped[int] = mapped_column(primary_key=True)
    handle: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class Ontology(Base, OwnedMixin, TimestampMixin):
    """A namespace. Every concept, group and category belongs to exactly one."""

    __tablename__ = "ontologies"
    __table_args__ = (UniqueConstraint("owner_id", "slug"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    concepts: Mapped[list[Concept]] = relationship(
        back_populates="ontology", cascade="all, delete-orphan"
    )
    categories: Mapped[list[Category]] = relationship(
        back_populates="ontology", cascade="all, delete-orphan"
    )
    groups: Mapped[list[ConceptGroup]] = relationship(
        back_populates="ontology", cascade="all, delete-orphan"
    )


class Category(Base, TimestampMixin):
    """Grouping label for concepts; drives per-category metric breakdowns."""

    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("ontology_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)

    ontology: Mapped[Ontology] = relationship(back_populates="categories")
    concepts: Mapped[list[Concept]] = relationship(back_populates="category")


class Concept(Base, TimestampMixin):
    """The unit that gets detected in a document.

    ``definition``, ``inclusion_criteria`` and ``exclusion_criteria`` are the text
    that reaches both the embedding model and the judge prompt, so edits to them
    change results and are what the snapshot layer versions.
    """

    __tablename__ = "concepts"
    __table_args__ = (UniqueConstraint("ontology_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    definition: Mapped[str | None] = mapped_column(Text)
    inclusion_criteria: Mapped[str | None] = mapped_column(Text)
    exclusion_criteria: Mapped[str | None] = mapped_column(Text)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    # Routing/scoring weight. Deliberately NOT part of the snapshot hash: it does
    # not change what the model is asked, so changing it must not invalidate
    # labels or fork the candidate artifact.
    weight: Mapped[float | None] = mapped_column(Float)

    ontology: Mapped[Ontology] = relationship(back_populates="concepts")
    category: Mapped[Category | None] = relationship(back_populates="concepts")


class ConceptGroup(Base, TimestampMixin):
    """A composite: several concepts together imply a higher-level situation."""

    __tablename__ = "concept_groups"

    id: Mapped[int] = mapped_column(primary_key=True)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # Set means this group is an editable fork of another.
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("concept_groups.id"))

    ontology: Mapped[Ontology] = relationship(back_populates="groups")
    parent: Mapped[ConceptGroup | None] = relationship(remote_side="ConceptGroup.id")
    members: Mapped[list[ConceptGroupMember]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class ConceptGroupMember(Base, TimestampMixin):
    """Weighted concept↔group edge.

    The weight lives on the edge, not the concept, so a fork tunes it locally.
    """

    __tablename__ = "concept_group_members"
    __table_args__ = (UniqueConstraint("group_id", "concept_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("concept_groups.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    weight: Mapped[float | None] = mapped_column(Float)

    group: Mapped[ConceptGroup] = relationship(back_populates="members")
    concept: Mapped[Concept] = relationship()


class Locus(Base, TimestampMixin):
    """The place dimension.

    A table rather than a bare string because feeds disagree on country coding —
    FIPS and ISO alpha-2 overlap with *different meanings* for some codes, so a
    naive string comparison silently mislabels events. The database is the single
    authority for the mapping and every layer resolves through it.
    """

    __tablename__ = "loci"

    id: Mapped[int] = mapped_column(primary_key=True)
    iso3: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)
    iso2: Mapped[str | None] = mapped_column(String(2), index=True)
    fips: Mapped[str | None] = mapped_column(String(2), index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
