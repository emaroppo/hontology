"""Which labels count: staleness, the trusted set, and what awaits a human.

**Staleness is per concept, not per version.** A label answers "does this document
evidence this concept, *as worded then*". Editing that wording invalidates the
answer. But editing one concept must not invalidate the whole bank, so staleness
is computed by comparing each concept's wording at the label's version against
its wording now — labels for untouched concepts stay valid even as the set's
version moves on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import TRUSTED_SOURCES, Concept, Observation, PairLabel
from hontology.evaluation.labels.bank import MACHINE
from hontology.ontology import snapshots


@dataclass
class LabelStats:
    total: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    positives: int = 0
    negatives: int = 0
    trusted: int = 0
    stale: int = 0
    pending_adjudication: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def stale_label_ids(session: Session, ontology_id: int) -> set[int]:
    """Labels whose concept has been reworded since they were made."""
    labels = list(
        session.scalars(
            select(PairLabel)
            .join(Concept, Concept.id == PairLabel.concept_id)
            .where(Concept.ontology_id == ontology_id)
        )
    )
    if not labels:
        return set()

    # One staleness computation per distinct version, not per label.
    by_version: dict[str, list[PairLabel]] = {}
    for label in labels:
        if label.ontology_version:
            by_version.setdefault(label.ontology_version, []).append(label)

    stale: set[int] = set()
    for version, group in by_version.items():
        changed = snapshots.stale_concept_ids(session, ontology_id, version)
        if not changed:
            continue
        stale.update(label.id for label in group if label.concept_id in changed)
    return stale


def trusted_labels(
    session: Session,
    ontology_id: int,
    *,
    include_machine: bool = False,
    include_stale: bool = False,
) -> list[PairLabel]:
    """The labels metrics should be computed over.

    Defaults exclude un-adjudicated machine labels and stale ones. Both
    exclusions are opt-out rather than opt-in, because both failure modes are
    silent: nothing about a stale label or an unreviewed machine label looks
    wrong at the point of use.
    """
    sources = set(TRUSTED_SOURCES) | ({MACHINE} if include_machine else set())
    labels = list(
        session.scalars(
            select(PairLabel)
            .join(Concept, Concept.id == PairLabel.concept_id)
            .where(Concept.ontology_id == ontology_id, PairLabel.source.in_(sources))
        )
    )
    if include_stale:
        return labels

    stale = stale_label_ids(session, ontology_id)
    return [label for label in labels if label.id not in stale]


def pending_adjudication(
    session: Session, ontology_id: int, *, limit: int = 50
) -> list[PairLabel]:
    """Machine proposals awaiting a human.

    These are the labels that do not yet count. Surfacing them is the difference
    between a design that gates on adjudication and one that merely says it does.
    """
    return list(
        session.scalars(
            select(PairLabel)
            .join(Concept, Concept.id == PairLabel.concept_id)
            .where(Concept.ontology_id == ontology_id, PairLabel.source == MACHINE)
            .order_by(PairLabel.id)
            .limit(limit)
        )
    )


def stale_labels(session: Session, ontology_id: int, *, limit: int = 50) -> list[PairLabel]:
    """Labels whose concept was reworded after they were made.

    Re-adjudicating one against the current wording makes it count again.
    """
    stale = stale_label_ids(session, ontology_id)
    if not stale:
        return []
    return list(
        session.scalars(
            select(PairLabel).where(PairLabel.id.in_(stale)).order_by(PairLabel.id).limit(limit)
        )
    )


def stats(session: Session, ontology_id: int) -> dict:
    """A summary an operator can act on."""
    rows = session.execute(
        select(PairLabel.source, PairLabel.matched, func.count(PairLabel.id))
        .join(Concept, Concept.id == PairLabel.concept_id)
        .where(Concept.ontology_id == ontology_id)
        .group_by(PairLabel.source, PairLabel.matched)
    ).all()

    summary = LabelStats()
    for source, matched, count in rows:
        summary.total += count
        summary.by_source[source] = summary.by_source.get(source, 0) + count
        if matched:
            summary.positives += count
        else:
            summary.negatives += count
        if source in TRUSTED_SOURCES:
            summary.trusted += count
        if source == MACHINE:
            summary.pending_adjudication += count

    summary.stale = len(stale_label_ids(session, ontology_id))
    summary.by_source = dict(sorted(summary.by_source.items()))

    observations = (
        session.scalar(
            select(func.count(Observation.id))
            .join(Concept, Concept.id == Observation.concept_id)
            .where(Concept.ontology_id == ontology_id)
        )
        or 0
    )
    return summary.as_dict() | {"observations": observations}
