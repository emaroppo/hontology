"""Ranking text a person pastes in, as a run's retrieval would rank an article.

The text is ranked against the leaves exactly as the chosen run's retrieval ranks
a stored article: the same embedding model and query prefix, the same class
wording and the same article length, then cut with a cutoff of choice. Nothing
about the text is stored: its embedding is computed, used and dropped.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from hontology.db.models import Run
from hontology.pipeline.retrieve import embed, model_text
from hontology.pipeline.retrieve.live import cut, leaf_texts, version_leaves
from hontology.pipeline.retrieve.tuning import Cutoff

_RANK = text(
    """
    SELECT object_id AS concept_id,
           1 - (embedding <=> CAST(:vector AS vector)) AS score
    FROM embeddings
    WHERE model_id = :model_id
      AND object_type = 'concept'
      AND object_id = ANY(:concept_ids)
      AND text_key = ANY(:concept_keys)
    ORDER BY embedding <=> CAST(:vector AS vector)
    LIMIT :pool_size
    """
)


def rank_text(
    session: Session,
    run: Run,
    body: str,
    cutoff: Cutoff,
    *,
    pool_size: int = 20,
) -> dict:
    """The leaves ranked against *body* as *run*'s retrieval would rank them."""
    if not body.strip():
        raise ValueError("there is no text to rank")
    version, source, settings, leaves = version_leaves(session, run)
    model = settings["embed_model"]
    provider = embed.get_provider(settings["embed_provider"])
    fields = settings["concept_fields"]
    items = leaf_texts(leaves, fields)
    # The run's own class embeddings: cached as usual, computed if missing.
    model_id, concept_keys = embed.ensure_embeddings(
        session,
        provider,
        model,
        object_type="concept",
        items={cid: text for cid, text in items.items() if text},
        fields=fields,
    )
    limit = settings["embed_body_limit"]
    vector = provider.embed(
        [model_text.document_prefix(model) + model_text.normalize_for_embedding(body[:limit])],
        model=model,
    )[0]
    pool = [
        (int(row.concept_id), float(row.score))
        for row in session.execute(
            _RANK,
            {
                "vector": "[" + ",".join(f"{x:.8f}" for x in vector) + "]",
                "model_id": model_id,
                "concept_ids": sorted(concept_keys),
                "concept_keys": sorted(set(concept_keys.values())),
                "pool_size": pool_size,
            },
        )
    ]
    kept = cut(pool, cutoff)
    names = {leaf["id"]: leaf["name"] for leaf in leaves}
    return {
        "version": version["version"],
        "source_run": source.id,
        "characters_used": min(len(body), limit),
        "characters_given": len(body),
        "ranking": [
            {
                "rank": rank,
                "concept_id": cid,
                "name": names.get(cid),
                "score": score,
                "kept": cid in kept,
            }
            for rank, (cid, score) in enumerate(pool, start=1)
        ],
    }
