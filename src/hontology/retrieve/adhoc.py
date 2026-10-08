"""Ranking text a person pastes in, as a run's retrieval would rank an article.

The corpus's article texts are not shipped, so the way to see retrieval at work
on something real is to paste an article in. The text is ranked against the
leaves exactly as the chosen run's retrieval ranks a stored article: the same
embedding model and query prefix, the same class wording (that of the run's
ontology version) and the same article length, then cut with a cutoff of
choice.

Nothing about the text is stored: it gets no document row, and its embedding
is computed, used and dropped. The classes' embeddings are cached as usual,
since they are the run's own.
"""

from __future__ import annotations

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Run
from hontology.evalkit import versions
from hontology.retrieve import embed
from hontology.retrieve.tuning import Cutoff, _cut

_RANK = sql(
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
    version = versions.retrieval_version(session, run)
    source = session.get(Run, version["source_run"])
    assert source is not None
    settings = version["settings"]
    model = settings["embed_model"]
    provider = embed.get_provider(settings["embed_provider"])
    leaves = versions.leaf_wording(session, source)
    fields = settings["concept_fields"]
    items = {
        leaf["id"]: embed.concept_text(
            Concept(
                name=leaf["name"],
                definition=leaf["definition"],
                inclusion_criteria=leaf["inclusion_criteria"],
            ),
            fields,
        )
        for leaf in leaves
    }
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
        [embed.document_prefix(model) + embed.normalize_for_embedding(body[:limit])],
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
    kept = _cut(pool, cutoff)
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
