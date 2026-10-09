"""Versions of a filter's code links.

Links are live rows a tick changes, so they are snapshotted when a run applies
them: otherwise nothing records which links a run fetched with.
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    Code,
    CodeSystem,
    Concept,
    ConceptCode,
    LinkSnapshot,
)
from hontology.pipeline.runs.config import stable_hash


def current_links(session: Session, ontology_id: int) -> list[list]:
    """``[[concept id, system slug, code], ...]``, sorted."""
    rows = session.execute(
        select(ConceptCode.concept_id, CodeSystem.slug, Code.code)
        .join(Code, Code.id == ConceptCode.code_id)
        .join(CodeSystem, CodeSystem.id == Code.system_id)
        .join(Concept, Concept.id == ConceptCode.concept_id)
        .where(Concept.ontology_id == ontology_id)
    )
    return sorted([concept_id, system, code] for concept_id, system, code in rows)


def resolve_links(session: Session, ontology_id: int) -> LinkSnapshot | None:
    """The snapshot of the live links, minting one if they changed. None when
    there are no links, since an ontology without them has no filter."""
    links = current_links(session, ontology_id)
    if not links:
        return None
    digest = stable_hash(links, 64)
    existing = session.scalar(
        select(LinkSnapshot).where(
            LinkSnapshot.ontology_id == ontology_id, LinkSnapshot.content_hash == digest
        )
    )
    if existing is not None:
        return existing
    count = len(
        list(
            session.scalars(
                select(LinkSnapshot.id).where(LinkSnapshot.ontology_id == ontology_id)
            )
        )
    )
    snapshot = LinkSnapshot(
        ontology_id=ontology_id,
        version=f"f{count + 1}",
        content_hash=digest,
        n_links=len(links),
        payload=json.dumps(links),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def link_snapshot(session: Session, ontology_id: int, version: str) -> list[list] | None:
    snapshot = session.scalar(
        select(LinkSnapshot).where(
            LinkSnapshot.ontology_id == ontology_id, LinkSnapshot.version == version
        )
    )
    return json.loads(snapshot.payload) if snapshot else None
