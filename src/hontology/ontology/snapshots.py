"""Change-triggered ontology snapshots.

A snapshot pins the *wording* of an ontology's concepts at a point in time. Two
things depend on it:

- a run records the version it executed against, so it can be replayed later
  against the definitions it was actually scored on;
- a ground-truth label records the version it was judged under, so editing a
  definition marks the label stale instead of silently invalidating it.

Snapshots are minted **only on change**. The set is hashed; if that hash is
already registered, the existing version is returned. Minting on every save would
produce a new version per keystroke and make the stamp meaningless.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, OntologySnapshot

# The fields whose text reaches the embedding model or the judge prompt. These,
# and only these, define a version.
#
# `weight` is excluded on purpose: it is a routing and scoring number that never
# reaches either, so changing it must not invalidate a label or fork a retrieval
# artifact. `category` is excluded for the same reason — it groups concepts for
# reporting but is not part of what the model is asked.
HASHED_FIELDS = ("id", "name", "definition", "inclusion_criteria", "exclusion_criteria")


@dataclass(frozen=True)
class SnapshotRef:
    version: str
    content_hash: str
    n_concepts: int
    created: bool  # True when this call minted it


def _normalize(concept: Concept | dict) -> dict[str, Any]:
    get: Callable[[str], Any] = (
        concept.get if isinstance(concept, dict) else lambda k: getattr(concept, k)
    )
    # Normalize whitespace so a trailing newline is not a new version, and treat
    # a cleared field as equivalent whether it arrives as None or "".
    row: dict[str, Any] = {"id": int(get("id"))}
    for field in HASHED_FIELDS[1:]:
        row[field] = (get(field) or "").strip()
    return row


def normalize_set(concepts: Sequence[Concept | dict]) -> list[dict[str, Any]]:
    """Concepts reduced to their hashed fields, ordered by id for stability."""
    return sorted((_normalize(c) for c in concepts), key=lambda r: r["id"])


def content_hash(concepts: Sequence[Concept | dict]) -> str:
    """Stable hash over the concept set's wording.

    Order-independent (rows are sorted by id) and whitespace-insensitive at the
    edges, so cosmetic edits do not mint versions.
    """
    canonical = json.dumps(
        normalize_set(concepts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_current(session: Session, ontology_id: int) -> SnapshotRef:
    """Resolve the live concept set to a snapshot, minting one only if changed."""
    concepts = list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
    digest = content_hash(concepts)

    existing = session.scalar(
        select(OntologySnapshot).where(
            OntologySnapshot.ontology_id == ontology_id,
            OntologySnapshot.content_hash == digest,
        )
    )
    if existing is not None:
        return SnapshotRef(
            version=existing.version,
            content_hash=existing.content_hash,
            n_concepts=existing.n_concepts,
            created=False,
        )

    count = (
        session.scalar(
            select(func.count(OntologySnapshot.id)).where(
                OntologySnapshot.ontology_id == ontology_id
            )
        )
        or 0
    )
    version = f"v{count + 1}"
    snapshot = OntologySnapshot(
        ontology_id=ontology_id,
        version=version,
        content_hash=digest,
        n_concepts=len(concepts),
        payload=json.dumps(normalize_set(concepts), ensure_ascii=False),
    )
    session.add(snapshot)
    session.flush()
    return SnapshotRef(
        version=version, content_hash=digest, n_concepts=len(concepts), created=True
    )


def get_version(session: Session, ontology_id: int, version: str) -> OntologySnapshot | None:
    return session.scalar(
        select(OntologySnapshot).where(
            OntologySnapshot.ontology_id == ontology_id,
            OntologySnapshot.version == version,
        )
    )


def resolve(session: Session, ontology_id: int, spec: str | None) -> SnapshotRef:
    """Resolve a config's version spec.

    ``None`` / ``"latest"`` resolves the live set (minting if it changed); an
    explicit ``"vN"`` must already exist, so a pinned replay can never silently
    drift onto newer wording.
    """
    if spec in (None, "", "latest"):
        return resolve_current(session, ontology_id)

    snapshot = get_version(session, ontology_id, str(spec))
    if snapshot is None:
        known = [
            s.version
            for s in session.scalars(
                select(OntologySnapshot)
                .where(OntologySnapshot.ontology_id == ontology_id)
                .order_by(OntologySnapshot.id)
            )
        ]
        raise ValueError(
            f"unknown ontology version {spec!r}; known: {', '.join(known) or '(none)'}"
        )
    return SnapshotRef(
        version=snapshot.version,
        content_hash=snapshot.content_hash,
        n_concepts=snapshot.n_concepts,
        created=False,
    )


def stale_concept_ids(session: Session, ontology_id: int, labeled_version: str) -> set[int]:
    """Concepts whose wording changed since *labeled_version*.

    A label stamped with that version is stale for exactly these concepts — it
    answered a question that is no longer being asked. Labels for concepts whose
    wording is unchanged remain valid even though the set's version moved on,
    which matters: editing one concept must not invalidate the whole bank.
    """
    snapshot = get_version(session, ontology_id, labeled_version)
    if snapshot is None:
        return set()

    then = {row["id"]: row for row in json.loads(snapshot.payload)}
    now = {
        row["id"]: row
        for row in normalize_set(
            list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
        )
    }

    stale: set[int] = set()
    for concept_id, current in now.items():
        previous = then.get(concept_id)
        # A concept that did not exist at snapshot time has no label to rot.
        if previous is not None and previous != current:
            stale.add(concept_id)
    return stale
