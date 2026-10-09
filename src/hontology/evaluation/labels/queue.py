"""Choosing which pairs are worth a human's attention.

Labelling is the scarce resource, so the queue is ranked by how much a label
teaches rather than by recency. In priority order:

**1. Cross-run disagreement.** Two runs that split on the same pair mean at least
one is wrong, and unlike confidence this signal does not depend on any model
being calibrated. It is the strongest thing available and it is free.

**2. Model uncertainty — but only from runs whose confidence means something.**
This is the subtle one. Many configurations emit degenerate confidence: always
≈0 or ≈1, or near-constant. A run that reports 0.5 on everything would look
maximally uncertain on every pair and flood the queue with noise, crowding out
the pairs that actually matter. So a run's confidence distribution is checked
before it is trusted as a signal, and ignored when it carries no information.

**3. Coverage.** Concepts with few labels, so the bank does not end up deep on
two concepts and empty on the rest.

Selection is then stratified with caps, because a purely greedy take-the-top-N
concentrates on whichever concept happens to be noisiest and produces a bank that
cannot support per-concept metrics.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.lookups import concepts_by_id
from hontology.db.models import Candidate, Concept, Document, PairLabel, Run, Verdict

# Thresholds for deciding whether a run's confidence carries usable signal.
CONF_MIN_SAMPLES = 20
CONF_MIN_STD = 0.05
CONF_INTERIOR_LO = 0.15
CONF_INTERIOR_HI = 0.85
CONF_MIN_INTERIOR_FRAC = 0.05


@dataclass
class QueueItem:
    document_id: int
    concept_id: int
    document_title: str
    document_url: str
    concept_name: str
    score: float
    reason: str
    verdicts: list[dict]
    proposed_matched: bool | None
    proposed_confidence: float | None

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "concept_id": self.concept_id,
            "document_title": self.document_title,
            "document_url": self.document_url,
            "concept_name": self.concept_name,
            "score": round(self.score, 4),
            "reason": self.reason,
            "verdicts": self.verdicts,
            "proposed_matched": self.proposed_matched,
            "proposed_confidence": self.proposed_confidence,
        }


def usable_confidence_runs(session: Session, run_ids: list[int]) -> set[int]:
    """Runs whose confidence distribution carries an uncertainty signal.

    Rejects too-few samples, near-constant output (a run that always says 0.5),
    and effectively binary output (a run that only ever says ~0 or ~1). Treating
    a degenerate distribution as uncertainty is worse than ignoring it: it
    actively misranks the queue.
    """
    usable: set[int] = set()
    for run_id in run_ids:
        values = [
            float(c)
            for (c,) in session.execute(
                select(Verdict.confidence).where(
                    Verdict.run_id == run_id,
                    Verdict.confidence.is_not(None),
                    Verdict.error.is_(None),
                )
            )
        ]
        if len(values) < CONF_MIN_SAMPLES:
            continue
        if statistics.pstdev(values) < CONF_MIN_STD:
            continue
        interior = sum(1 for v in values if CONF_INTERIOR_LO < v < CONF_INTERIOR_HI) / len(
            values
        )
        if interior < CONF_MIN_INTERIOR_FRAC:
            continue
        usable.add(run_id)
    return usable


def _labelled_pairs(session: Session, ontology_id: int) -> set[tuple[int, int]]:
    return {
        (document_id, concept_id)
        for document_id, concept_id in session.execute(
            select(PairLabel.document_id, PairLabel.concept_id)
            .join(Concept, Concept.id == PairLabel.concept_id)
            .where(Concept.ontology_id == ontology_id)
        )
    }


def _label_counts(session: Session, ontology_id: int) -> dict[int, int]:
    return {
        concept_id: count
        for concept_id, count in session.execute(
            select(PairLabel.concept_id, func.count(PairLabel.id))
            .join(Concept, Concept.id == PairLabel.concept_id)
            .where(Concept.ontology_id == ontology_id)
            .group_by(PairLabel.concept_id)
        )
    }


def build_queue(
    session: Session,
    ontology_id: int,
    *,
    run_ids: list[int] | None = None,
    limit: int = 50,
    per_concept_cap: int | None = None,
    include_unjudged: bool = True,
) -> list[QueueItem]:
    """Rank unlabelled pairs by how informative a label would be."""
    if run_ids is None:
        run_ids = [
            r.id
            for r in session.scalars(
                select(Run).where(Run.ontology_id == ontology_id).order_by(Run.id)
            )
        ]
    if not run_ids:
        return []

    usable = usable_confidence_runs(session, run_ids)
    already = _labelled_pairs(session, ontology_id)
    label_counts = _label_counts(session, ontology_id)

    # pair -> [{run, matched, confidence}]
    by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for verdict in session.scalars(
        select(Verdict).where(Verdict.run_id.in_(run_ids), Verdict.error.is_(None))
    ):
        if verdict.matched is None:
            continue
        by_pair[(verdict.document_id, verdict.concept_id)].append(
            {
                "run_id": verdict.run_id,
                "matched": bool(verdict.matched),
                "confidence": verdict.confidence,
                "evidence": verdict.evidence,
                "confidence_usable": verdict.run_id in usable,
            }
        )

    if include_unjudged:
        # Candidates that retrieval selected but no run ever judged. These are a
        # blind spot: they carry no verdict at all, so nothing else surfaces them.
        for candidate in session.scalars(
            select(Candidate).where(Candidate.run_id.in_(run_ids), Candidate.selected.is_(True))
        ):
            by_pair.setdefault((candidate.document_id, candidate.concept_id), [])

    concepts = concepts_by_id(session, ontology_id)
    max_labels = max(label_counts.values(), default=0) or 1

    items: list[QueueItem] = []
    for (document_id, concept_id), verdicts in by_pair.items():
        if (document_id, concept_id) in already or concept_id not in concepts:
            continue

        matched_values = {v["matched"] for v in verdicts}
        disagreement = len(matched_values) > 1

        informative = [
            v for v in verdicts if v["confidence_usable"] and v["confidence"] is not None
        ]
        # Distance from the decision boundary, only from runs worth believing.
        uncertainty = (
            max(0.0, 1.0 - 2 * abs(statistics.mean(v["confidence"] for v in informative) - 0.5))
            if informative
            else 0.0
        )
        coverage = 1.0 - (label_counts.get(concept_id, 0) / max_labels)

        if disagreement:
            score, reason = 1.0 + uncertainty, "runs disagree"
        elif not verdicts:
            score, reason = 0.5 + 0.3 * coverage, "never judged"
        elif uncertainty > 0:
            score, reason = 0.4 * uncertainty + 0.2 * coverage, "model uncertain"
        else:
            score, reason = 0.1 * coverage, "coverage"

        document = session.get(Document, document_id)
        if document is None:
            continue

        proposed = verdicts[0] if verdicts else None
        items.append(
            QueueItem(
                document_id=document_id,
                concept_id=concept_id,
                document_title=document.title or "",
                document_url=document.url,
                concept_name=concepts[concept_id].name,
                score=score,
                reason=reason,
                verdicts=verdicts,
                proposed_matched=proposed["matched"] if proposed else None,
                proposed_confidence=proposed["confidence"] if proposed else None,
            )
        )

    items.sort(key=lambda i: i.score, reverse=True)
    if per_concept_cap is None:
        return items[:limit]

    # Stratify: a purely greedy take-the-top-N concentrates on whichever concept
    # is noisiest, leaving a bank that cannot support per-concept metrics.
    taken: dict[int, int] = defaultdict(int)
    chosen: list[QueueItem] = []
    for item in items:
        if len(chosen) >= limit:
            break
        if taken[item.concept_id] >= per_concept_cap:
            continue
        chosen.append(item)
        taken[item.concept_id] += 1

    # Backfill if the caps left the queue short.
    if len(chosen) < limit:
        seen = {(i.document_id, i.concept_id) for i in chosen}
        for item in items:
            if len(chosen) >= limit:
                break
            if (item.document_id, item.concept_id) not in seen:
                chosen.append(item)
    return chosen
