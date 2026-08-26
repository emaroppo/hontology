"""Ingest endpoints.

These are all **on-demand**. The API never starts a scheduler: continuous ingest
is a separate opt-in process (`hontology ingest watch`), so importing this router
cannot cause anything to start fetching.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hontology.db.session import get_db
from hontology.ingest import scrape as scrape_service
from hontology.ingest import service

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestStatusOut(BaseModel):
    feed: str
    watermark: str | None
    # None when nothing has been ingested. Deliberately not 0, which would make a
    # fresh install look healthy and current.
    lag_slices: int | None
    slice_counts: dict[str, int]
    documents: int
    documents_unfetched: int


class CatchUpIn(BaseModel):
    max_slices: int = 32


class BackfillIn(BaseModel):
    start: str
    end: str


@router.get("/status", response_model=IngestStatusOut)
def ingest_status(db: Session = Depends(get_db)):
    """Watermark, lag and slice outcomes — whether the feed is being kept up with."""
    return service.status(db)


@router.post("/catch-up")
def catch_up(payload: CatchUpIn | None = None, db: Session = Depends(get_db)):
    """Process everything between the watermark and the newest published slice."""
    try:
        return service.catch_up(db, max_slices=(payload or CatchUpIn()).max_slices)
    except Exception as exc:  # noqa: BLE001 - reported rather than a 500 traceback
        raise HTTPException(
            http_status.HTTP_502_BAD_GATEWAY, f"catch-up failed: {exc}"
        ) from exc


@router.post("/backfill")
def backfill(payload: BackfillIn, db: Session = Depends(get_db)):
    """Ingest an explicit historical window without moving the watermark."""
    try:
        return service.backfill(db, payload.start, payload.end)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            http_status.HTTP_502_BAD_GATEWAY, f"backfill failed: {exc}"
        ) from exc


class ScrapeIn(BaseModel):
    limit: int | None = None
    retry_failed: bool = False
    # The third-party reader service sends target URLs to an external host, so it
    # stays opt-in per request rather than becoming a silent fallback.
    reader_proxy: bool = False


@router.post("/scrape")
def scrape(payload: ScrapeIn | None = None, db: Session = Depends(get_db)):
    """Fetch article text for documents that do not have it yet."""
    body = payload or ScrapeIn()
    return scrape_service.scrape_pending(
        db,
        limit=body.limit,
        retry_failed=body.retry_failed,
        use_reader_proxy=body.reader_proxy,
    )
