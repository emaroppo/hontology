"""Evaluation endpoints.

One router per section, included in this order: routes match in the order
they were added.
"""

from __future__ import annotations

from fastapi import APIRouter

from hontology.apps.api.routers.evaluation import diagnostics, live, pairs, sample

router = APIRouter()
router.include_router(pairs.router)
router.include_router(sample.router)
router.include_router(diagnostics.router)
router.include_router(live.router)
