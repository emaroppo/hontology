"""Near-duplicate concepts: stored vectors too close to tell apart."""

from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Embedding, EmbeddingModel
from hontology.ontology.lint.finding import Finding

DEFAULT_DUPLICATE_THRESHOLD = 0.92


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def check_near_duplicates(
    session: Session,
    ontology_id: int,
    *,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    embed_model: str | None = None,
) -> list[Finding]:
    """Concepts whose stored vectors are too close to tell apart.

    Uses whatever vectors already exist rather than embedding on demand, so lint
    stays free. Returns nothing when the ontology has never been embedded.
    """
    concepts = {
        c.id: c
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    if len(concepts) < 2:
        return []

    query = select(Embedding).where(
        Embedding.object_type == "concept", Embedding.object_id.in_(concepts)
    )
    if embed_model:
        model_row = session.scalar(
            select(EmbeddingModel).where(EmbeddingModel.model_name == embed_model)
        )
        if model_row is None:
            return []
        query = query.where(Embedding.model_id == model_row.id)

    # Keep one vector per concept — the most recent, which reflects current wording.
    vectors: dict[int, list[float]] = {}
    for row in session.scalars(query.order_by(Embedding.id)):
        vectors[row.object_id] = list(row.embedding)

    findings: list[Finding] = []
    ids = sorted(vectors)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            score = _cosine(vectors[left], vectors[right])
            if score < threshold:
                continue
            findings.append(
                Finding(
                    check="near_duplicate",
                    severity="warning",
                    concept_id=left,
                    concept_name=concepts[left].name,
                    related_concept_id=right,
                    score=round(score, 4),
                    message=(
                        f"embeds almost identically to {concepts[right].name!r} "
                        f"(cosine {score:.3f}). Retrieval cannot separate them, so "
                        "labels for one contaminate the other and per-concept "
                        "metrics stop meaning anything"
                    ),
                )
            )
    return findings
