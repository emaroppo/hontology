"""Retrieval ranked afresh against a version's leaves, rather than read from a run.

Two uses. A retrieval version is scored on every labelled article, ranked from
cached embeddings only: nothing is embedded, so scoring stays free. And text a
person pastes in is ranked as the chosen run's retrieval would rank an article:
the same embedding model and query prefix, the same class wording (that of the
run's ontology version) and the same article length, then cut with a cutoff of
choice. The corpus's article texts are not shipped, so that is the way to see
retrieval at work on something real.

Nothing about pasted text is stored: it gets no document row, and its
embedding is computed, used and dropped. The classes' embeddings are cached as
usual, since they are the run's own.
"""

from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from hontology.db.models import Concept, EmbeddingModel, Run
from hontology.evalkit import versions
from hontology.evalkit.metrics import wilson
from hontology.retrieve import embed
from hontology.retrieve.candidates import select_adaptive
from hontology.retrieve.tuning import Cutoff, Truth


def _version_leaves(session: Session, run: Run) -> tuple[dict, Run, dict, list[dict]]:
    """A run's retrieval version, its source run, settings and leaf wording."""
    version = versions.retrieval_version(session, run)
    source = session.get(Run, version["source_run"])
    assert source is not None
    return version, source, version["settings"], versions.leaf_wording(session, source)


def _leaf_texts(leaves: list[dict], fields: str) -> dict[int, str]:
    """Each leaf's text as the version embedded it."""
    return {
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


# Every labelled article ranked against the version's leaves, from cached
# embeddings only: nothing is embedded here.
_LIVE_RANKING = text(
    """
    WITH docs AS (
        SELECT DISTINCT ON (object_id) object_id AS document_id, embedding
        FROM embeddings
        WHERE model_id = :model_id
          AND object_type = 'document'
          AND object_id = ANY(:documents)
          AND text_key LIKE :document_prefix
        ORDER BY object_id, id DESC
    ),
    ranked AS (
        SELECT d.document_id, c.object_id AS concept_id,
               1 - (c.embedding <=> d.embedding) AS score,
               ROW_NUMBER() OVER (
                   PARTITION BY d.document_id ORDER BY c.embedding <=> d.embedding
               ) AS rank
        FROM docs d
        CROSS JOIN embeddings c
        WHERE c.model_id = :model_id
          AND c.object_type = 'concept'
          AND c.object_id = ANY(:concept_ids)
          AND c.text_key = ANY(:concept_keys)
    )
    SELECT document_id, concept_id, score, rank FROM ranked WHERE rank <= :pool_size
    """
)


def _cut(pool: list[tuple[int, float]], cutoff: Cutoff) -> set[int]:
    if cutoff.selection == "top-k":
        return {cid for cid, _ in pool[: cutoff.top_k]}
    return {
        cid
        for cid, _ in select_adaptive(
            pool, min_score=cutoff.min_score, rel_margin=cutoff.rel_margin, max_k=cutoff.max_k
        )
    }


def live_report(
    session: Session, run: Run, cutoff: Cutoff, labels: Truth, *, pool_size: int = 20
) -> dict:
    """A retrieval version scored on every labelled article, ranked afresh.

    *run* is any run with the version; its source run supplies the settings and
    the leaves' wording. Articles whose embedding under these settings is not
    cached are counted, not embedded: scoring stays free.
    """
    version, _, settings, leaves = _version_leaves(session, run)
    leaf_ids = {leaf["id"] for leaf in leaves}
    fields = settings["concept_fields"]
    concept_keys = [
        embed.content_key(fields, text) for text in _leaf_texts(leaves, fields).values()
    ]
    model = session.scalar(
        select(EmbeddingModel).where(
            EmbeddingModel.key == f"{settings['embed_provider']}/{settings['embed_model']}"
        )
    )
    truth = {pair: matched for pair, matched in labels.items() if pair[1] in leaf_ids}
    labelled = sorted({doc for doc, _ in truth})

    pools: dict[int, list[tuple[int, float]]] = {}
    if model is not None and labelled:
        for row in session.execute(
            _LIVE_RANKING,
            {
                "model_id": model.id,
                "documents": labelled,
                "document_prefix": f"body@{settings['embed_body_limit']}@%",
                "concept_ids": sorted(leaf_ids),
                "concept_keys": concept_keys,
                "pool_size": pool_size,
            },
        ):
            pools.setdefault(row.document_id, []).append((row.concept_id, float(row.score)))
    for pool in pools.values():
        pool.sort(key=lambda pair: -pair[1])

    ranked = set(pools)
    positives = {pair for pair, matched in truth.items() if matched and pair[0] in ranked}
    in_pool = {(doc, cid) for doc, pool in pools.items() for cid, _ in pool}
    kept = {(doc, cid) for doc, pool in pools.items() for cid in _cut(pool, cutoff)}
    kept_labelled = [pair for pair in kept if pair in truth]
    kept_positive = sum(1 for pair in kept_labelled if truth[pair])
    found = len(positives & kept)
    low, high = wilson(found, len(positives))
    pool_found = len(positives & in_pool)
    pool_low, pool_high = wilson(pool_found, len(positives))
    per_class: dict[int, list[int]] = {}
    for doc, cid in positives:
        counts = per_class.setdefault(cid, [0, 0, 0])
        counts[0] += 1
        counts[1] += (doc, cid) in in_pool
        counts[2] += (doc, cid) in kept
    names = {leaf["id"]: leaf["name"] for leaf in leaves}
    rank_of = {
        (doc, cid): position
        for doc, pool in pools.items()
        for position, (cid, _) in enumerate(pool, start=1)
    }
    recall_at_k = {
        k: (
            sum(1 for pair in positives if rank_of.get(pair, pool_size + 1) <= k)
            / len(positives)
        )
        if positives
        else None
        for k in range(1, pool_size + 1)
    }
    return {
        "version": version,
        "recall_at_k": recall_at_k,
        "cutoff": asdict(cutoff),
        "pool_size": pool_size,
        "documents": {
            "labelled": len(labelled),
            "ranked": len(ranked),
            "not_embedded": len(labelled) - len(ranked),
        },
        "pairs_kept": len(kept),
        "pairs_per_document": len(kept) / len(ranked) if ranked else None,
        "positives": len(positives),
        "recall": {
            "found": found,
            "rate": found / len(positives) if positives else None,
            "ci": [low, high],
        },
        "pool_recall": {
            "found": pool_found,
            "rate": pool_found / len(positives) if positives else None,
            "ci": [pool_low, pool_high],
        },
        "lost_to_cutoff": len(positives & in_pool) - found,
        "never_ranked": len(positives) - pool_found,
        "kept_labelled": len(kept_labelled),
        "kept_positive": kept_positive,
        "per_class": sorted(
            (
                {"class": names.get(cid, str(cid)), "positives": n, "in_pool": p, "kept": k}
                for cid, (n, p, k) in per_class.items()
            ),
            key=lambda row: (row["kept"] - row["positives"], row["class"]),
        ),
    }


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
    version, source, settings, leaves = _version_leaves(session, run)
    model = settings["embed_model"]
    provider = embed.get_provider(settings["embed_provider"])
    fields = settings["concept_fields"]
    items = _leaf_texts(leaves, fields)
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
