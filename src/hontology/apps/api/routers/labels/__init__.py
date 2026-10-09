"""Ground-truth bank endpoints: the labelling queue, labels, observations.

One router per section, included in this order: routes match in the order
they were added.
"""

from __future__ import annotations

from fastapi import APIRouter

from hontology.apps.api.routers.labels import bank, documents, portability

router = APIRouter()
router.include_router(bank.router)
router.include_router(portability.router)
router.include_router(documents.router)
