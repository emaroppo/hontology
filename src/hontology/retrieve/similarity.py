"""Concept ↔ code similarity, and the curated links it proposes.

A run is fully described by (embedding model, which concept fields were embedded,
which code level and text were compared). Every pairwise score is persisted —
including the ones below threshold — so a proposal stays auditable afterwards and
a threshold can be re-chosen without recomputing anything.

**The invariant that makes curation survivable:** an automatically proposed link
carries the id of the run that proposed it; a link a human added by hand has
NULL. Recomputation deletes and rebuilds only the former. Without that split, a
recompute silently discards every hand-made correction, which is exactly the kind
of loss nobody notices until the curation is long gone.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, insert, select, text
from sqlalchemy.orm import Session

from hontology.db.models import (
    Code,
    CodeSystem,
    Concept,
    ConceptCode,
    SimilarityRun,
    SimilarityScore,
)
from hontology.ingest import themes
from hontology.retrieve.candidates import select_adaptive
from hontology.retrieve.embed import (
    EmbeddingProvider,
    embed_concepts,
    ensure_embeddings,
)

# Cosine similarity via pgvector's distance operator. Vectors are normalized, so
# 1 - distance is the cosine similarity.
_PAIRWISE = text(
    """
    SELECT ce.object_id AS concept_id,
           kc.object_id AS code_id,
           1 - (ce.embedding <=> kc.embedding) AS score
    FROM embeddings ce
    JOIN embeddings kc ON kc.model_id = ce.model_id
    WHERE ce.model_id = :model_id
      AND ce.object_type = 'concept'
      AND ce.text_key = ANY(:concept_keys)
      AND kc.object_type = 'code'
      AND kc.text_key = ANY(:code_keys)
    """
)


@dataclass(frozen=True)
class SimilarityResult:
    run_id: int
    n_scores: int
    n_linked: int
    n_manual_preserved: int


def embed_codes(
    session: Session,
    provider: EmbeddingProvider,
    model: str,
    *,
    system_id: int,
    level: str,
) -> tuple[int, dict[int, str]]:
    """Embed the proposable codes of one level."""
    return ensure_embeddings(
        session,
        provider,
        model,
        object_type="code",
        items=proposal_texts(session, system_id=system_id, level=level),
        fields=f"code:{level}",
    )


def proposal_texts(session: Session, *, system_id: int, level: str) -> dict[int, str]:
    """``{code id: text to embed}`` for the codes a run may propose.

    A CAMEO code is its description. GKG themes are filtered and made readable
    first (see `themes.proposal_text`): most are entity lists, not events.
    """
    system = session.get(CodeSystem, system_id)
    codes = session.scalars(
        select(Code).where(Code.system_id == system_id, Code.level == level)
    )
    out: dict[int, str] = {}
    for code in codes:
        if system is not None and system.slug == themes.THEMES_SLUG:
            text = themes.proposal_text(code.code, code.name)
        else:
            text = (code.name or code.code).strip()
        if text:
            out[code.id] = text
    return out


def run_similarity(
    session: Session,
    provider: EmbeddingProvider,
    model: str,
    *,
    ontology_id: int,
    system_id: int,
    level: str,
    concept_fields: str = "name+definition",
    threshold: float = 0.45,
    adaptive: bool = False,
    rel_margin: float = 0.10,
    min_score: float = 0.25,
    max_k: int = 15,
    auto_link: bool = True,
    notes: str | None = None,
) -> SimilarityResult:
    """Score every concept against every code at *level*, and propose links."""
    model_id, concept_keys = embed_concepts(
        session, provider, model, ontology_id, fields=concept_fields
    )
    _, code_keys = embed_codes(session, provider, model, system_id=system_id, level=level)

    run = SimilarityRun(
        model_id=model_id,
        source_object_type="concept",
        source_text_key=concept_fields,
        target_object_type=f"code:{level}",
        target_text_key=f"code:{level}",
        metric="cosine",
        threshold=threshold,
        notes=notes,
    )
    session.add(run)
    session.flush()

    pairs = session.execute(
        _PAIRWISE,
        {
            "model_id": model_id,
            "concept_keys": list(set(concept_keys.values())),
            "code_keys": list(set(code_keys.values())),
        },
    ).all()

    if pairs:
        session.execute(
            insert(SimilarityScore),
            [
                {
                    "run_id": run.id,
                    "source_id": int(p.concept_id),
                    "target_id": int(p.code_id),
                    "score": float(p.score),
                }
                for p in pairs
            ],
        )
        session.flush()

    n_linked = 0
    n_manual = 0
    if auto_link:
        selected = (
            _select_adaptive(pairs, min_score=min_score, rel_margin=rel_margin, max_k=max_k)
            if adaptive
            else _select_threshold(pairs, threshold)
        )
        n_linked, n_manual = _apply_links(
            session, run.id, ontology_id, system_id, level, selected
        )

    return SimilarityResult(
        run_id=run.id, n_scores=len(pairs), n_linked=n_linked, n_manual_preserved=n_manual
    )


def _select_threshold(pairs, threshold: float) -> dict[tuple[int, int], float]:
    best: dict[tuple[int, int], float] = {}
    for p in pairs:
        score = float(p.score)
        if score < threshold:
            continue
        key = (int(p.concept_id), int(p.code_id))
        if score > best.get(key, -1.0):
            best[key] = score
    return best


def _select_adaptive(
    pairs, *, min_score: float, rel_margin: float, max_k: int
) -> dict[tuple[int, int], float]:
    """Per-concept selection: keep codes within *rel_margin* of that concept's best.

    A flat threshold suits concepts unevenly — a concept whose best match scores
    0.8 and one whose best scores 0.5 need different cutoffs, and a single number
    either floods the first with weak links or leaves the second with none.
    """
    by_concept: dict[int, dict[int, float]] = {}
    for p in pairs:
        by_concept.setdefault(int(p.concept_id), {})[int(p.code_id)] = float(p.score)

    selected: dict[tuple[int, int], float] = {}
    for concept_id, targets in by_concept.items():
        ranked = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)
        for code_id, score in select_adaptive(
            ranked, min_score=min_score, rel_margin=rel_margin, max_k=max_k
        ):
            selected[(concept_id, code_id)] = score
    return selected


def _apply_links(
    session: Session,
    run_id: int,
    ontology_id: int,
    system_id: int,
    level: str,
    selected: dict[tuple[int, int], float],
) -> tuple[int, int]:
    """Replace auto-proposed links for this level, leaving manual ones untouched."""
    concept_ids = {
        c.id for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    level_code_ids = {
        c.id
        for c in session.scalars(
            select(Code).where(Code.system_id == system_id, Code.level == level)
        )
    }

    # Only auto-proposed rows (run id set) for this ontology and level are cleared.
    session.execute(
        delete(ConceptCode).where(
            ConceptCode.similarity_run_id.is_not(None),
            ConceptCode.concept_id.in_(concept_ids),
            ConceptCode.code_id.in_(level_code_ids),
        )
    )
    session.flush()

    manual = {
        (row.concept_id, row.code_id)
        for row in session.scalars(
            select(ConceptCode).where(
                ConceptCode.concept_id.in_(concept_ids),
                ConceptCode.code_id.in_(level_code_ids),
            )
        )
    }

    added = 0
    for (concept_id, code_id), score in selected.items():
        # Never duplicate a pair a human already asserted by hand.
        if (concept_id, code_id) in manual:
            continue
        session.add(
            ConceptCode(
                concept_id=concept_id,
                code_id=code_id,
                similarity_score=score,
                similarity_run_id=run_id,
            )
        )
        added += 1
    session.flush()
    return added, len(manual)


def candidates_for_concept(
    session: Session, run_id: int, concept_id: int, *, limit: int = 20
) -> list[tuple[Code, float]]:
    """Top scoring codes for a concept from one run, for the review UI."""
    rows = session.execute(
        select(SimilarityScore.target_id, SimilarityScore.score)
        .where(SimilarityScore.run_id == run_id, SimilarityScore.source_id == concept_id)
        .order_by(SimilarityScore.score.desc())
        .limit(limit)
    ).all()
    codes = {
        c.id: c for c in session.scalars(select(Code).where(Code.id.in_([r[0] for r in rows])))
    }
    return [(codes[r[0]], float(r[1])) for r in rows if r[0] in codes]


def set_link(session: Session, concept_id: int, code_id: int, *, linked: bool) -> None:
    """Tick or untick an association by hand.

    A hand-made link is stored with a NULL run id, which is what protects it from
    the next recompute.
    """
    existing = session.scalar(
        select(ConceptCode).where(
            ConceptCode.concept_id == concept_id, ConceptCode.code_id == code_id
        )
    )
    if linked and existing is None:
        session.add(ConceptCode(concept_id=concept_id, code_id=code_id))
    elif not linked and existing is not None:
        session.delete(existing)
    elif linked and existing is not None:
        # Confirming a proposal makes it manual, so a recompute cannot drop it.
        existing.similarity_run_id = None
    session.flush()


def import_links(session: Session, ontology_id: int, rows: list[dict]) -> dict:
    """Apply hand-curated links from ``{concept, system, code}`` rows.

    Keyed by concept name, system slug and code string, never ids, so a
    curation file moves between databases. Every row is resolved before any is
    written: a typo in one code must not leave half a mapping applied.
    """
    concepts = {
        c.name: c.id
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    codes = {
        (system, code): code_id
        for system, code, code_id in session.execute(
            select(CodeSystem.slug, Code.code, Code.id).join(
                CodeSystem, CodeSystem.id == Code.system_id
            )
        )
    }
    problems = [
        f"{row['concept']} -> {row['system']}:{row['code']}"
        for row in rows
        if row["concept"] not in concepts or (row["system"], row["code"]) not in codes
    ]
    if problems:
        raise LookupError("unresolved links: " + "; ".join(problems))

    for row in rows:
        set_link(
            session, concepts[row["concept"]], codes[(row["system"], row["code"])], linked=True
        )
    return {"links": len(rows), "concepts": len({row["concept"] for row in rows})}
