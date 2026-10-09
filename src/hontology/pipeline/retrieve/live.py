"""Scoring a retrieval version on every labelled article, ranked afresh.

Each article is ranked against the version's leaves from cached embeddings only:
nothing is embedded, so scoring stays free. The helpers that resolve a version's
leaves and their text are shared with `pasted`, which ranks text a person pastes in.
"""

from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from hontology.db.models import Concept, EmbeddingModel, Run
from hontology.evaluation.metrics.intervals import wilson
from hontology.pipeline.retrieve import embed
from hontology.pipeline.retrieve.candidates import select_adaptive
from hontology.pipeline.retrieve.tuning import Cutoff, Truth
from hontology.pipeline.runs import versions


def version_leaves(session: Session, run: Run) -> tuple[dict, Run, dict, list[dict]]:
    """A run's retrieval version, its source run, settings and leaf wording."""
    version = versions.retrieval_version(session, run)
    source = session.get(Run, version["source_run"])
    assert source is not None
    return version, source, version["settings"], versions.leaf_wording(session, source)


def leaf_texts(leaves: list[dict], fields: str) -> dict[int, str]:
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


def cut(pool: list[tuple[int, float]], cutoff: Cutoff) -> set[int]:
    if cutoff.selection == "top-k":
        return {cid for cid, _ in pool[: cutoff.top_k]}
    return {
        cid
        for cid, _ in select_adaptive(
            pool, min_score=cutoff.min_score, rel_margin=cutoff.rel_margin, max_k=cutoff.max_k
        )
    }


def _live_pools(
    session: Session, settings: dict, leaves: list[dict], labelled: list[int], pool_size: int
) -> dict[int, list[tuple[int, float]]]:
    """Each labelled article's pool, best first, from cached embeddings only."""
    fields = settings["concept_fields"]
    concept_keys = [
        embed.content_key(fields, text) for text in leaf_texts(leaves, fields).values()
    ]
    model = session.scalar(
        select(EmbeddingModel).where(
            EmbeddingModel.key == f"{settings['embed_provider']}/{settings['embed_model']}"
        )
    )
    pools: dict[int, list[tuple[int, float]]] = {}
    if model is not None and labelled:
        for row in session.execute(
            _LIVE_RANKING,
            {
                "model_id": model.id,
                "documents": labelled,
                "document_prefix": f"body@{settings['embed_body_limit']}@%",
                "concept_ids": sorted(leaf["id"] for leaf in leaves),
                "concept_keys": concept_keys,
                "pool_size": pool_size,
            },
        ):
            pools.setdefault(row.document_id, []).append((row.concept_id, float(row.score)))
    for pool in pools.values():
        pool.sort(key=lambda pair: -pair[1])
    return pools


def _recall_at_k(
    positives: set[tuple[int, int]], pools: dict[int, list[tuple[int, float]]], pool_size: int
) -> dict[int, float | None]:
    rank_of = {
        (doc, cid): position
        for doc, pool in pools.items()
        for position, (cid, _) in enumerate(pool, start=1)
    }
    return {
        k: (
            sum(1 for pair in positives if rank_of.get(pair, pool_size + 1) <= k)
            / len(positives)
        )
        if positives
        else None
        for k in range(1, pool_size + 1)
    }


def _per_class(
    positives: set[tuple[int, int]],
    in_pool: set[tuple[int, int]],
    kept: set[tuple[int, int]],
    leaves: list[dict],
) -> list[dict]:
    """Per class: its positives, how many the pool reached and the cutoff kept."""
    per_class: dict[int, list[int]] = {}
    for doc, cid in positives:
        counts = per_class.setdefault(cid, [0, 0, 0])
        counts[0] += 1
        counts[1] += (doc, cid) in in_pool
        counts[2] += (doc, cid) in kept
    names = {leaf["id"]: leaf["name"] for leaf in leaves}
    return sorted(
        (
            {"class": names.get(cid, str(cid)), "positives": n, "in_pool": p, "kept": k}
            for cid, (n, p, k) in per_class.items()
        ),
        key=lambda row: (row["kept"] - row["positives"], row["class"]),
    )


def _found(found: int, positives: int) -> dict:
    low, high = wilson(found, positives)
    return {"found": found, "rate": found / positives if positives else None, "ci": [low, high]}


def live_report(
    session: Session, run: Run, cutoff: Cutoff, labels: Truth, *, pool_size: int = 20
) -> dict:
    """A retrieval version scored on every labelled article, ranked afresh.

    *run* is any run with the version; its source run supplies the settings and
    the leaves' wording. Articles whose embedding under these settings is not
    cached are counted, not embedded: scoring stays free.
    """
    version, _, settings, leaves = version_leaves(session, run)
    leaf_ids = {leaf["id"] for leaf in leaves}
    truth = {pair: matched for pair, matched in labels.items() if pair[1] in leaf_ids}
    labelled = sorted({doc for doc, _ in truth})
    pools = _live_pools(session, settings, leaves, labelled, pool_size)

    ranked = set(pools)
    positives = {pair for pair, matched in truth.items() if matched and pair[0] in ranked}
    in_pool = {(doc, cid) for doc, pool in pools.items() for cid, _ in pool}
    kept = {(doc, cid) for doc, pool in pools.items() for cid in cut(pool, cutoff)}
    kept_labelled = [pair for pair in kept if pair in truth]
    found = len(positives & kept)
    pool_found = len(positives & in_pool)
    return {
        "version": version,
        "recall_at_k": _recall_at_k(positives, pools, pool_size),
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
        "recall": _found(found, len(positives)),
        "pool_recall": _found(pool_found, len(positives)),
        "lost_to_cutoff": len(positives & in_pool) - found,
        "never_ranked": len(positives) - pool_found,
        "kept_labelled": len(kept_labelled),
        "kept_positive": sum(1 for pair in kept_labelled if truth[pair]),
        "per_class": _per_class(positives, in_pool, kept, leaves),
    }
