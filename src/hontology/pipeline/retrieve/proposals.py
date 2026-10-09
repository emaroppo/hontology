"""Which scored concept-code pairs become proposed links, and writing them.

Proposals are rebuilt on every similarity run; links a person made by hand are
never touched (see `retrieve.similarity`).
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from hontology.db.models import Code, Concept, ConceptCode
from hontology.pipeline.retrieve.candidates import select_adaptive


def select_by_threshold(pairs, threshold: float) -> dict[tuple[int, int], float]:
    best: dict[tuple[int, int], float] = {}
    for p in pairs:
        score = float(p.score)
        if score < threshold:
            continue
        key = (int(p.concept_id), int(p.code_id))
        if score > best.get(key, -1.0):
            best[key] = score
    return best


def select_per_concept(
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


def apply_links(
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
