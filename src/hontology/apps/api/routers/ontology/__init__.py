"""Ontology endpoints.

The UI reaches every one of these over HTTP and never touches the database, which
is what keeps the Streamlit layer replaceable without moving any logic.

One router per section, included in this order: routes match in the order
they were added.
"""

from __future__ import annotations

from fastapi import APIRouter

from hontology.apps.api.routers.ontology import (
    concepts,
    ontologies,
    portability,
    structure,
    versioning,
)

router = APIRouter()
router.include_router(ontologies.router)
router.include_router(concepts.router)
router.include_router(structure.router)
router.include_router(portability.router)
router.include_router(versioning.router)
