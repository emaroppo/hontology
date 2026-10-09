"""The ingest watermark: the newest slice up to which a feed is contiguous.

It advances **contiguously**: to a slice only when that slice is the immediate
successor of the current watermark, so a slice that failed leaves the watermark
behind it and is retried on the next pass.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import IngestWatermark
from hontology.pipeline.ingest.feed import gdelt


def get_watermark(session: Session, feed: str) -> IngestWatermark:
    row = session.scalar(select(IngestWatermark).where(IngestWatermark.feed == feed))
    if row is None:
        row = IngestWatermark(feed=feed)
        session.add(row)
        session.flush()
    return row


def advance_watermark(session: Session, key: str, feed: str) -> bool:
    """Move the watermark to *key* if it is the next contiguous slice.

    Returns whether it moved. Refusing to skip ahead is what stops a successful
    later slice from marking an earlier failed one as done.
    """
    mark = get_watermark(session, feed)
    if mark.last_slice_key is not None and key != gdelt.next_key(mark.last_slice_key):
        return False
    mark.last_slice_key = key
    mark.last_sliced_at = gdelt.key_to_datetime(key)
    session.flush()
    return True
