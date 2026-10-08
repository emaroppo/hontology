"""FastAPI application.

This process owns every database connection in the system. The UI, the CLI and
any future client all go through here, so there is exactly one place where
persistence rules live.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hontology.api.routers import (
    evaluation,
    ingest,
    judgement,
    labels,
    ontology,
    retrieval,
    runs,
    taxonomy,
)
from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.ingest.loci import seed_loci


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings().ensure_dirs()
    # Country codes are reference data, not user content: seeding them keeps the
    # "empty on first run" promise about the ontology while making sure the
    # FIPS↔ISO authority exists before any feed row needs resolving.
    with session_scope() as session:
        seed_loci(session)
    yield


app = FastAPI(
    title="hontology",
    version="0.1.0",
    summary="Define an ontology, detect it in a live news feed, evaluate the result.",
    lifespan=lifespan,
)

app.include_router(ontology.router)
app.include_router(taxonomy.router)
app.include_router(ingest.router)
app.include_router(labels.router)
app.include_router(evaluation.router)
app.include_router(runs.router)
app.include_router(retrieval.router)
app.include_router(judgement.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
