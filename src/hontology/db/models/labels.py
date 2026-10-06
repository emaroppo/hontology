"""The ground truth bank.

Two tiers, deliberately **independent of each other**. This is the single most
important structural decision in the project, and it was arrived at the hard way,
so the reasoning is recorded here rather than left implicit.

**`Observation`** — "concept C occurred at locus L on date D." Positives only:
there is no such thing as a negative observation, because "this did not happen
anywhere on this date" is not a claim anyone can label. Observations are the
recall denominator, and they are independent of any pipeline run — they stay
valid when the retrieval strategy, the model, or the prompt changes.

**`PairLabel`** — "document D does / does not evidence concept C." Both
polarities, and crucially **no foreign key to `Observation`**.

The missing foreign key is the entire point. In the first version of this schema
a pair label was a child of an event, which meant a label could only exist by
first asserting that an event occurred. That made the most valuable label in the
whole system — *the pipeline proposed this pair and it was wrong* — literally
unrepresentable, because a rejected match has no event to hang from. Precision
could not be measured. Detaching the tiers fixed it.

An observation's supporting documents are therefore *derived*, by querying
positive pair labels matching its ``(concept, locus, date)``, rather than stored.
"""

from __future__ import annotations

from datetime import date as _date

from sqlalchemy import Boolean, Date, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, TimestampMixin
from hontology.db.models.ontology import Concept, Locus

# Which label sources count as ground truth by default.
#
# `machine` is excluded: scoring an LLM judge against labels another LLM produced
# unsupervised measures inter-model agreement, not correctness. A human
# confirming a machine proposal promotes it to `adjudicated`, which does count —
# that promotion step is what breaks the circularity.
TRUSTED_SOURCES: frozenset[str] = frozenset({"human", "adjudicated", "imported"})
ALL_SOURCES: frozenset[str] = TRUSTED_SOURCES | {"machine"}


class Observation(Base, TimestampMixin):
    """A known occurrence. Positive by construction."""

    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("concept_id", "locus_id", "occurred_on"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    concept_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    locus_id: Mapped[int] = mapped_column(
        ForeignKey("loci.id", ondelete="CASCADE"), nullable=False, index=True
    )
    occurred_on: Mapped[_date] = mapped_column(Date, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    source_note: Mapped[str | None] = mapped_column(Text)

    concept: Mapped[Concept] = relationship()
    locus: Mapped[Locus] = relationship()


class PairLabel(Base, TimestampMixin):
    """A standalone judgment about one (document, concept) pair.

    ``ontology_version`` stamps the snapshot the concept's wording was at when
    this judgment was made. When the definition later changes, this label is
    *stale* — it answered a different question — and the metrics layer drops it
    from the default denominator instead of silently scoring against it.
    """

    __tablename__ = "pair_labels"
    __table_args__ = (UniqueConstraint("document_id", "concept_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    matched: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Where the locus/date claim sits, when the labeler asserted one. Nullable:
    # a pure negative ("this article does not evidence C") needs neither.
    locus_id: Mapped[int | None] = mapped_column(ForeignKey("loci.id", ondelete="SET NULL"))
    occurred_on: Mapped[_date | None] = mapped_column(Date, index=True)

    # "human" | "adjudicated" | "machine" | "imported" — see TRUSTED_SOURCES.
    source: Mapped[str] = mapped_column(String, nullable=False, default="human", index=True)
    # Set when source == "machine" or "adjudicated": who proposed it.
    proposed_by: Mapped[str | None] = mapped_column(String)
    note: Mapped[str | None] = mapped_column(Text)

    ontology_version: Mapped[str | None] = mapped_column(String, index=True)

    concept: Mapped[Concept] = relationship()
    locus: Mapped[Locus | None] = relationship()


class CalendarReview(Base, TimestampMixin):
    """A person's answer to "is this matched article about that calendar entry?".

    For an event or precursor: does the article describe *this* event, rather
    than another of the same kind? For a control: does it report a real instance
    of the concept there and then, which would make the control itself wrong?

    Keyed by the calendar entry's id and the article's URL, never database ids,
    so one review serves every run and arm that matches the same article.
    """

    __tablename__ = "calendar_reviews"
    __table_args__ = (UniqueConstraint("entry_id", "document_url"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    document_url: Mapped[str] = mapped_column(Text, nullable=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
