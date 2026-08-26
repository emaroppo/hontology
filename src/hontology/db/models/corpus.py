"""Ingested feed slices and the document store.

Every part of the scrape cache design is there to prevent a specific failure:

- **Body text lives on disk, not in the database.** The row is an index. Bodies
  are large, rarely queried by content, and cheap to store as files.
- **The file is the source of truth.** If the indexed file is missing, the row is
  treated as a miss and the URL is re-fetched. This keeps a half-deleted cache
  self-healing instead of silently serving empty bodies.
- **Failures are recorded as rows**, with an empty ``body_path`` and an error. A
  dead or paywalled URL produces no file, so without a row every run would hit it
  over the network again. The row is what makes a permanent failure cost one
  request ever, with an explicit opt-in to retry.
- **Paths are relative** to the store directory, so the cache survives a move.

`Document` is keyed by URL and shared across every slice, config and run: the
expensive network step is never repeated because something downstream changed.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hontology.db.base import Base, TimestampMixin


class FeedSlice(Base, TimestampMixin):
    """One unit of ingest — for GDELT v2, one 15-minute export file.

    Status is recorded per slice so a gap in the corpus is *visible*. A skipped
    or failed slice that leaves no row is indistinguishable from a quiet period,
    which makes downstream recall numbers quietly wrong.
    """

    __tablename__ = "feed_slices"
    __table_args__ = (UniqueConstraint("feed", "slice_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    feed: Mapped[str] = mapped_column(String, nullable=False, default="gdelt_v2", index=True)
    # For GDELT v2: the "YYYYMMDDHHMMSS" stamp of the export file.
    slice_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sliced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # "pending" | "ok" | "empty" | "failed"
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)

    rows_in_feed: Mapped[int] = mapped_column(Integer, default=0)
    rows_matched: Mapped[int] = mapped_column(Integer, default=0)
    documents_new: Mapped[int] = mapped_column(Integer, default=0)
    documents_cached: Mapped[int] = mapped_column(Integer, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float | None] = mapped_column()


class IngestWatermark(Base, TimestampMixin):
    """Last slice successfully processed per feed.

    Catch-up runs forward from here; an explicit backfill window deliberately
    does not move it, so a historical backfill cannot make the ingester think it
    is up to date and skip the live gap.
    """

    __tablename__ = "ingest_watermarks"

    id: Mapped[int] = mapped_column(primary_key=True)
    feed: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    last_slice_key: Mapped[str | None] = mapped_column(String)
    last_sliced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Document(Base, TimestampMixin):
    """One article. The row is an index; the body is a file on disk."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    # sha1(url)[:16] — the body filename stem, and a short stable handle.
    url_hash: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(Text)
    # Relative to the configured scrape-cache directory. Empty for a failure.
    body_path: Mapped[str | None] = mapped_column(Text)
    body_chars: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[str | None] = mapped_column(String(8))

    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    http_status: Mapped[int | None] = mapped_column(Integer)
    extractor: Mapped[str | None] = mapped_column(String)
    # Recorded so a library upgrade that changes extraction is attributable
    # rather than showing up as an unexplained metric shift.
    extractor_version: Mapped[str | None] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text)

    # Set at ingest, not discovered later as a mysterious judge failure.
    is_junk: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    junk_reason: Mapped[str | None] = mapped_column(String)

    @property
    def ok(self) -> bool:
        return bool(self.body_path) and not self.error


class FeedEvent(Base):
    """A structured event row from the feed, retained for the code-based filter.

    Only the columns the pipeline actually uses are kept; the feed's full width is
    left in the downloaded file rather than mirrored into the database.
    """

    __tablename__ = "feed_events"
    __table_args__ = (UniqueConstraint("slice_id", "feed_event_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    slice_id: Mapped[int] = mapped_column(
        ForeignKey("feed_slices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    feed_event_id: Mapped[str] = mapped_column(String, nullable=False)
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )

    event_code: Mapped[str | None] = mapped_column(String, index=True)
    base_code: Mapped[str | None] = mapped_column(String, index=True)
    root_code: Mapped[str | None] = mapped_column(String, index=True)
    locus_id: Mapped[int | None] = mapped_column(ForeignKey("loci.id", ondelete="SET NULL"))
    occurred_on: Mapped[str | None] = mapped_column(String(10), index=True)

    document: Mapped[Document | None] = relationship()
