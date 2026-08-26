"""FastAPI application.

This process owns every database connection in the system. The UI, the CLI and
any future client all go through here, so there is exactly one place where
persistence rules live.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hontology.api.routers import ontology
from hontology.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings().ensure_dirs()
    yield


app = FastAPI(
    title="hontology",
    version="0.1.0",
    summary="Define an ontology, detect it in a live news feed, evaluate the result.",
    lifespan=lifespan,
)

app.include_router(ontology.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
