"""Candidate selection.

Concepts are ranked against each document body by embedding similarity.

An earlier design also offered a *code* source, reaching concepts through the
feed's own CAMEO codes. It was removed: as a retrieval strategy it competed with
semantic similarity and lost, and it only worked for ontologies that map onto
CAMEO at all. That same signal is genuinely valuable one stage earlier, though —
see `ingest.filter`, where codes decide which documents are worth fetching, a
question semantic similarity cannot answer because it needs the body first.

**The full pre-cutoff pool is stored**, with ``selected`` marking what survived.
That extra column separates two questions a single number conflates:

    did retrieval rank the right concept highly?   — measured over the pool
    did the cutoff keep it?                        — measured over `selected`

A run that loses recall at the cutoff needs a different fix from one whose
embedding never surfaced the concept at all, and without the pool you cannot tell
which you are looking at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Document, Embedding
from hontology.ontology import hierarchy
from hontology.retrieve import embed

log = logging.getLogger(__name__)


@dataclass
class CandidateStats:
    documents: int = 0
    pool_rows: int = 0
    selected_rows: int = 0
    embedded: int = 0
    embeddings_reused: int = 0

    def as_dict(self) -> dict:
        return {
            "documents": self.documents,
            "pool_rows": self.pool_rows,
            "selected_rows": self.selected_rows,
            "embedded": self.embedded,
            "embeddings_reused": self.embeddings_reused,
        }


def _embedding_count(session: Session, model_id: int, object_type: str) -> int:
    return (
        session.scalar(
            select(func.count(Embedding.id)).where(
                Embedding.model_id == model_id, Embedding.object_type == object_type
            )
        )
        or 0
    )


def select_adaptive(
    ranked: list[tuple[int, float]], *, min_score: float, rel_margin: float, max_k: int
) -> list[tuple[int, float]]:
    """Keep concepts within *rel_margin* of this document's best score.

    A flat threshold suits documents unevenly: one whose best match scores 0.8
    and one whose best scores 0.5 need different cutoffs, and a single number
    either floods the first or starves the second.
    """
    if not ranked:
        return []
    cutoff = max(min_score, ranked[0][1] - rel_margin)
    return [(cid, score) for cid, score in ranked if score >= cutoff][:max_k]


def document_body(document: Document, limit: int) -> str | None:
    settings = get_settings()
    if not document.body_path:
        return None
    path = settings.scrape_cache_dir / document.body_path
    if not path.exists():
        # The file is the source of truth; a missing one means no usable body.
        return None
    return path.read_text(encoding="utf-8")[:limit]


def _clear_run(session: Session, run_id: int) -> None:
    session.execute(delete(Candidate).where(Candidate.run_id == run_id))
    session.flush()


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------

# The document vector is looked up rather than passed in, so a cached embedding
# never has to make the round trip back into Python.
_NEAREST_CONCEPTS = text(
    """
    WITH doc AS (
        SELECT embedding
        FROM embeddings
        WHERE model_id = :model_id
          AND object_type = 'document'
          AND object_id = :document_id
          AND text_key = :document_key
        LIMIT 1
    )
    SELECT c.object_id AS concept_id,
           1 - (c.embedding <=> doc.embedding) AS score
    FROM embeddings c, doc
    WHERE c.model_id = :model_id
      AND c.object_type = 'concept'
      AND c.object_id = ANY(:concept_ids)
      AND c.text_key = ANY(:text_keys)
    ORDER BY c.embedding <=> doc.embedding
    LIMIT :pool_size
    """
)


def build_semantic(
    session: Session,
    run_id: int,
    *,
    ontology_id: int,
    documents: list[Document],
    config: dict,
    embed_body_limit: int,
    refresh_embeddings: bool = False,
) -> CandidateStats:
    """Rank concepts against each document body by cosine similarity."""
    provider = embed.get_provider(config["embed_provider"])
    model = config["embed_model"]

    model_id, concept_keys = embed.embed_concepts(
        session, provider, model, ontology_id, fields=config["concept_fields"]
    )
    # Leaves only: internal classes of a hierarchy are never retrieved, so a flat
    # run is unchanged when an ontology gains structure. In a flat ontology every
    # concept is a leaf.
    leaf_ids = hierarchy.leaves(session, ontology_id)
    concept_keys = {cid: key for cid, key in concept_keys.items() if cid in leaf_ids}
    if not concept_keys:
        return CandidateStats()
    text_keys = list(set(concept_keys.values()))
    # Matched by concept id as well as text key: a text key alone is shared by
    # any concept with identical wording, including deleted ones and other
    # ontologies' copies.
    concept_ids = sorted(concept_keys)

    stats = CandidateStats()

    # Embed every body once, reusing anything already cached. Only the missing
    # ones cost a provider call.
    bodies: dict[int, str] = {}
    for document in documents:
        body = document_body(document, embed_body_limit)
        if body:
            bodies[document.id] = body
    if not bodies:
        return stats

    before = _embedding_count(session, model_id, "document")
    _, document_keys = embed.embed_documents(
        session,
        provider,
        model,
        bodies,
        body_limit=embed_body_limit,
        refresh=refresh_embeddings,
    )
    stats.embedded = _embedding_count(session, model_id, "document") - before
    stats.embeddings_reused = len(bodies) - stats.embedded

    for document in documents:
        if document.id not in document_keys:
            continue

        pool = [
            (int(row[0]), float(row[1]))
            for row in session.execute(
                _NEAREST_CONCEPTS,
                {
                    "model_id": model_id,
                    "document_id": document.id,
                    "document_key": document_keys[document.id],
                    "text_keys": text_keys,
                    "concept_ids": concept_ids,
                    "pool_size": config["pool_size"],
                },
            ).all()
        ]
        if not pool:
            continue

        chosen = (
            select_adaptive(
                pool,
                min_score=config["min_score"],
                rel_margin=config["rel_margin"],
                max_k=config["max_k"],
            )
            if config["selection"] == "adaptive"
            else pool[: config["top_k"]]
        )
        chosen_ids = {cid for cid, _ in chosen}

        for rank, (concept_id, score) in enumerate(pool, start=1):
            session.add(
                Candidate(
                    run_id=run_id,
                    document_id=document.id,
                    concept_id=concept_id,
                    source="semantic",
                    score=score,
                    rank=rank,
                    selected=concept_id in chosen_ids,
                )
            )
        stats.documents += 1
        stats.pool_rows += len(pool)
        stats.selected_rows += len(chosen_ids)

    session.flush()
    return stats


def build_candidates(
    session: Session,
    run_id: int,
    *,
    ontology_id: int,
    documents: list[Document],
    config: dict,
    embed_body_limit: int,
    refresh_embeddings: bool = False,
) -> dict:
    """Build candidates for the run, replacing any previous rows."""
    _clear_run(session, run_id)

    stats = build_semantic(
        session,
        run_id,
        ontology_id=ontology_id,
        documents=documents,
        config=config,
        embed_body_limit=embed_body_limit,
        refresh_embeddings=refresh_embeddings,
    )

    log.info("candidates (%s): %s", config["source"], stats.as_dict())
    return stats.as_dict() | {"source": config["source"]}
