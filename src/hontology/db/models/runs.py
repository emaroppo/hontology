"""Runs, candidates and verdicts.

`Run` doubles as the job record for the async API: `POST /runs` inserts one and
returns immediately, the worker advances its status, and `GET /runs/{id}` reads
it back. Keeping the job state in the same row as the run's identity means a
process restart loses progress reporting but never loses the run.

`Candidate` stores the **full pre-cutoff pool** with a ``selected`` flag rather
than only the survivors. That one extra column is what separates two questions
that otherwise get conflated:

    did retrieval rank the right concept highly?   (measured over the pool)
    did the cutoff keep it?                        (measured over `selected`)

A run that loses recall at the cutoff needs a different fix from one whose
embedding never surfaced the concept at all, and without the pool you cannot tell
which you have.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, OwnedMixin, TimestampMixin


class Run(Base, OwnedMixin, TimestampMixin):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    ontology_id: Mapped[int] = mapped_column(
        ForeignKey("ontologies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ontology_version: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # The normalized, fully-defaulted run config as stored.
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Infrastructure actually used (hosts, URLs, library versions). Recorded for
    # provenance; deliberately absent from every stage key.
    manifest: Mapped[dict | None] = mapped_column(JSONB)

    # Composed per-stage keys. Two runs sharing candidates_key share the
    # retrieval artifact and it is not recomputed.
    candidates_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    judge_key: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # "pending" | "running" | "done" | "failed" | "cancelled"
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    stage: Mapped[str | None] = mapped_column(String)
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Candidate(Base):
    """One (document, concept) pair proposed by retrieval.

    Rows are written for the whole pool; ``selected`` marks those that survived
    the cutoff and were sent to the judge.
    """

    __tablename__ = "candidates"
    __table_args__ = (UniqueConstraint("run_id", "document_id", "concept_id"),)

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

    # "code" | "semantic"
    source: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # For the code source: which code actually matched, and at what granularity.
    matched_code: Mapped[str | None] = mapped_column(String)
    matched_level: Mapped[str | None] = mapped_column(String)


class Verdict(Base, TimestampMixin):
    """One judged pair.

    ``confidence`` is the model's self-report for a single sample, but is
    overwritten by the **vote fraction** when a run takes several samples. The
    vote fraction is the more honest number: a model asked five times and
    answering "yes" three times is genuinely uncertain in a way its own stated
    0.9 does not capture.
    """

    __tablename__ = "verdicts"
    __table_args__ = (UniqueConstraint("run_id", "document_id", "concept_id"),)

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

    matched: Mapped[bool | None] = mapped_column(Boolean, index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    vote_fraction: Mapped[float | None] = mapped_column(Float)
    locus_id: Mapped[int | None] = mapped_column(ForeignKey("loci.id", ondelete="SET NULL"))
    evidence: Mapped[str | None] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text)

    provider: Mapped[str | None] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    prompt_id: Mapped[str | None] = mapped_column(String, index=True)
    mode: Mapped[str | None] = mapped_column(String)
    samples: Mapped[int] = mapped_column(Integer, default=1)

    latency_s: Mapped[float | None] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    # A recorded error means the pair was attempted and failed. Distinguishing
    # that from "never attempted" is what the liveness gate checks.
    error: Mapped[str | None] = mapped_column(Text)

    run: Mapped[Run] = relationship()


class ExtractedEvent(Base, TimestampMixin):
    """One event a judge found in an article, for runs that judge events.

    An extract-then-classify run first lists an article's distinct events, then
    gives each at most one leaf class, so one event cannot be spread across
    sibling classes while an article reporting several events keeps them all.
    The article's verdicts are derived from its events; the events are kept as
    the evidence for them, and as the first form of the event records that
    dates, phases and relations will later attach to.
    """

    __tablename__ = "extracted_events"
    __table_args__ = (UniqueConstraint("run_id", "document_id", "ordinal"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str | None] = mapped_column(Text)
    # happened | threatened | ended, as the extraction call reported it.
    status: Mapped[str | None] = mapped_column(String)
    country: Mapped[str | None] = mapped_column(String(2))
    # The leaf it was classified as, if any; parents it was routed through.
    concept_id: Mapped[int | None] = mapped_column(
        ForeignKey("concepts.id", ondelete="SET NULL"), index=True
    )
    routed_through: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, default=list
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    # classified | unclassified (weighed, no leaf fitted) | rejected (only
    # "other" accepted it at the top) | error
    outcome: Mapped[str | None] = mapped_column(String, index=True)
    error: Mapped[str | None] = mapped_column(Text)
