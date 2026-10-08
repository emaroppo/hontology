"""Re-applying a different cutoff to a finished run's stored pool.

Every run stores its whole ranked pool before the cutoff, so the cutoff's
settings can be changed after the fact without embedding anything: the same
ranking, a different line through it. That is what tuning needs, because the
cutoff decides how many pairs reach the judge (its cost) and which true matches
never do (recall lost before judging can start).

The kept set is computed in SQL, because a run's pool is hundreds of thousands
of rows. It is the same rule as `candidates.select_adaptive`: the pool is ranked
by score, so "within the margin of the document's best, at most max-k" is a
score test and a rank test.

Recall is measured against labels on the run's documents (human labels from
the bank, or a machine annotation set), split the way
`candidates` stores it: a true match missing from the pool was never ranked high
enough, one in the pool but cut was lost to the cutoff. Those need different
fixes, so they are reported apart.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Document, EmbeddingModel, Run
from hontology.evalkit import config as run_config
from hontology.ontology import hierarchy
from hontology.retrieve import embed
from hontology.retrieve.candidates import document_body, select_adaptive

SELECTIONS = ("adaptive", "top-k")


@dataclass(frozen=True)
class Cutoff:
    selection: str = "adaptive"
    top_k: int = 3
    min_score: float = 0.45
    rel_margin: float = 0.05
    max_k: int = 8

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"unknown selection {self.selection!r}")


def run_cutoff(run: Run) -> Cutoff:
    """The cutoff a run was actually built with."""
    candidates = run_config.normalize(run.config or {})["candidates"]
    return Cutoff(
        selection=candidates["selection"],
        top_k=candidates["top_k"],
        min_score=candidates["min_score"],
        rel_margin=candidates["rel_margin"],
        max_k=candidates["max_k"],
    )


# Each pool row with whether *cutoff* keeps it. `rank` is 1-based by score.
_POOL = text(
    """
    SELECT document_id, concept_id, score, rank,
           CASE WHEN :selection = 'top-k' THEN rank <= :top_k
                ELSE score >= GREATEST(:min_score,
                                       MAX(score) OVER (PARTITION BY document_id) - :rel_margin)
                     AND rank <= :max_k
           END AS kept
    FROM candidates
    WHERE run_id = :run_id
    """
)

_KEPT_PER_DOCUMENT = text(
    f"""
    SELECT kept_here, COUNT(*) FROM (
        SELECT document_id, COUNT(*) FILTER (WHERE kept) AS kept_here
        FROM ({_POOL.text}) pool
        GROUP BY document_id
    ) per_document
    GROUP BY kept_here
    """
)


def _params(run_id: int, cutoff: Cutoff) -> dict:
    return {"run_id": run_id} | asdict(cutoff)


Truth = dict[tuple[int, int], bool]


def truth(session: Session, ontology_id: int, annotator: str | None = None) -> Truth:
    """``(document, concept) -> matched``: the human label bank (trusted,
    current), or the machine annotation set *annotator*. See
    `evalkit.annotations`."""
    from hontology.evalkit import annotations

    return annotations.truth(session, ontology_id, annotator)


def report(session: Session, run_id: int, cutoff: Cutoff, labels: Truth) -> dict:
    """What *cutoff* keeps over the run's pool, and what it costs in recall."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")

    histogram = {
        int(kept): int(n)
        for kept, n in session.execute(_KEPT_PER_DOCUMENT, _params(run_id, cutoff))
    }
    documents = sum(histogram.values())
    pairs = sum(kept * n for kept, n in histogram.items())

    labelled_documents = {document_id for document_id, _ in labels}
    # Retrieval ranks leaves only, so labels on internal classes are not its to find.
    leaves = hierarchy.leaves(session, run.ontology_id)
    in_pool: set[tuple[int, int]] = set()
    kept: set[tuple[int, int]] = set()
    run_documents: set[int] = set()
    if labelled_documents:
        rows = session.execute(
            text(f"SELECT * FROM ({_POOL.text}) pool WHERE document_id = ANY(:documents)"),
            _params(run_id, cutoff) | {"documents": sorted(labelled_documents)},
        )
        for row in rows:
            pair = (row.document_id, row.concept_id)
            run_documents.add(row.document_id)
            in_pool.add(pair)
            if row.kept:
                kept.add(pair)

    # Only labels on documents this run retrieved for, and on classes it ranks.
    truth = {
        pair: matched
        for pair, matched in labels.items()
        if pair[0] in run_documents and pair[1] in leaves
    }
    positives = {pair for pair, matched in truth.items() if matched}
    kept_labelled = [pair for pair in kept if pair in truth]
    kept_positive = sum(1 for pair in kept_labelled if truth[pair])

    return {
        "cutoff": asdict(cutoff),
        "documents": documents,
        "pairs_kept": pairs,
        "pairs_per_document": pairs / documents if documents else None,
        "documents_with_none": histogram.get(0, 0),
        "kept_per_document": dict(sorted(histogram.items())),
        "labels": {
            "documents": len(run_documents),
            "positives": len(positives),
            "positives_in_pool": len(positives & in_pool),
            "positives_kept": len(positives & kept),
            "kept_labelled": len(kept_labelled),
            "kept_positive": kept_positive,
        },
    }


def document_pool(
    session: Session, run_id: int, document_id: int, cutoff: Cutoff, labels: Truth
) -> dict:
    """One article's ranked classes, with what the run and *cutoff* keep."""
    run = session.get(Run, run_id)
    document = session.get(Document, document_id)
    if run is None or document is None:
        raise LookupError(f"run {run_id} or document {document_id} does not exist")
    names: dict[int, str] = {
        concept_id: name
        for concept_id, name in session.execute(
            select(Concept.id, Concept.name).where(Concept.ontology_id == run.ontology_id)
        )
    }
    rows = session.execute(
        text(
            f"SELECT p.*, c.selected FROM ({_POOL.text}) p "
            "JOIN candidates c ON c.run_id = :run_id AND c.document_id = p.document_id "
            "AND c.concept_id = p.concept_id WHERE p.document_id = :document_id ORDER BY p.rank"
        ),
        _params(run_id, cutoff) | {"document_id": document_id},
    )
    pool = [
        {
            "concept_id": row.concept_id,
            "name": names.get(row.concept_id),
            "score": row.score,
            "rank": row.rank,
            "selected": row.selected,
            "kept": bool(row.kept),
            "label": labels.get((document_id, row.concept_id)),
        }
        for row in rows
    ]
    ranked = {row["concept_id"] for row in pool}
    leaves = hierarchy.leaves(session, run.ontology_id)
    body = document_body(document, 4000) or ""
    return {
        "document_id": document.id,
        "url": document.url,
        "title": document.title or body.strip().split("\n", 1)[0][:160] or None,
        "body": body,
        "pool": pool,
        # Labelled positives the pool never ranked: no cutoff can recover them.
        "missed": sorted(
            names[concept_id]
            for (doc, concept_id), matched in labels.items()
            if doc == document_id
            and matched
            and concept_id in leaves
            and concept_id not in ranked
        ),
    }


def concept_documents(
    session: Session,
    run_id: int,
    concept_id: int,
    cutoff: Cutoff,
    labels: Truth,
    *,
    limit: int = 50,
) -> list[dict]:
    """The articles that rank a class highest, with what the run and *cutoff* keep."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")
    rows = session.execute(
        text(
            f"SELECT p.*, c.selected, d.url, d.title FROM ({_POOL.text}) p "
            "JOIN candidates c ON c.run_id = :run_id AND c.document_id = p.document_id "
            "AND c.concept_id = p.concept_id "
            "JOIN documents d ON d.id = p.document_id "
            "WHERE p.concept_id = :concept_id ORDER BY p.score DESC LIMIT :limit"
        ),
        _params(run_id, cutoff) | {"concept_id": concept_id, "limit": limit},
    )
    return [
        {
            "document_id": row.document_id,
            "url": row.url,
            "title": row.title,
            "score": row.score,
            "rank": row.rank,
            "selected": row.selected,
            "kept": bool(row.kept),
            "label": labels.get((row.document_id, concept_id)),
        }
        for row in rows
    ]


def labelled_documents(session: Session, run_id: int, labels: Truth) -> list[dict]:
    """The run's articles that carry trusted labels, most positives first."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")
    positives = Counter(doc for (doc, _), matched in labels.items() if matched)
    labelled = {doc for doc, _ in labels}
    in_run = set(
        session.scalars(
            text(
                "SELECT DISTINCT document_id FROM candidates "
                "WHERE run_id = :run_id AND document_id = ANY(:documents)"
            ),
            {"run_id": run_id, "documents": sorted(labelled)},
        )
    )
    documents = {
        d.id: d for d in session.scalars(select(Document).where(Document.id.in_(in_run)))
    }
    return sorted(
        (
            {
                "document_id": doc,
                "url": documents[doc].url,
                "title": documents[doc].title,
                "positives": positives.get(doc, 0),
            }
            for doc in in_run
        ),
        key=lambda d: (-d["positives"], d["document_id"]),
    )


# ---------------------------------------------------------------------------
# Live scoring of a retrieval version
# ---------------------------------------------------------------------------

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
    from hontology.evalkit import versions
    from hontology.evalkit.metrics import wilson

    version = versions.retrieval_version(session, run)
    source = session.get(Run, version["source_run"])
    assert source is not None
    settings = version["settings"]
    leaves = versions.leaf_wording(session, source)
    leaf_ids = {leaf["id"] for leaf in leaves}
    concept_keys = [
        embed.content_key(
            settings["concept_fields"],
            embed.concept_text(
                Concept(
                    name=leaf["name"],
                    definition=leaf["definition"],
                    inclusion_criteria=leaf["inclusion_criteria"],
                ),
                settings["concept_fields"],
            ),
        )
        for leaf in leaves
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
