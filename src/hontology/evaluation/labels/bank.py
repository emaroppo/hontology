"""The ground-truth bank: writing labels and observations.

Two tiers, independent of each other (see `db.models.labels` for why the absent
foreign key is load-bearing). This module writes to them; `labels.staleness`
decides which of their labels can be trusted.

**Provenance gates the denominator.** A machine-proposed label does not count as
ground truth until a human has confirmed or flipped it, at which point it becomes
`adjudicated`. Scoring an LLM judge against labels another LLM produced
unsupervised measures agreement between two models, not correctness — the human
step is what breaks the loop, so the default denominator excludes un-adjudicated
machine labels rather than quietly folding them in.
"""

from __future__ import annotations

from datetime import date as _date

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    TRUSTED_SOURCES,
    Concept,
    Document,
    Observation,
    PairLabel,
)
from hontology.ontology import snapshots

MACHINE = "machine"
ADJUDICATED = "adjudicated"
HUMAN = "human"
IMPORTED = "imported"


def upsert_label(
    session: Session,
    *,
    document_id: int,
    concept_id: int,
    matched: bool,
    source: str = HUMAN,
    locus_id: int | None = None,
    occurred_on: _date | None = None,
    note: str | None = None,
    proposed_by: str | None = None,
    ontology_id: int | None = None,
) -> PairLabel:
    """Create or update the label for one pair.

    The label is stamped with the ontology version current *now*, which is what
    later makes staleness detectable.
    """
    concept = session.get(Concept, concept_id)
    if concept is None:
        raise LookupError(f"concept {concept_id} does not exist")

    version = snapshots.resolve_current(session, ontology_id or concept.ontology_id).version

    label = session.scalar(
        select(PairLabel).where(
            PairLabel.document_id == document_id, PairLabel.concept_id == concept_id
        )
    )
    if label is None:
        label = PairLabel(document_id=document_id, concept_id=concept_id, matched=matched)
        session.add(label)

    label.matched = matched
    label.source = source
    label.locus_id = locus_id
    label.occurred_on = occurred_on
    label.note = note
    label.proposed_by = proposed_by
    label.ontology_version = version
    session.flush()
    return label


def adjudicate(
    session: Session, label_id: int, *, matched: bool, note: str | None = None
) -> PairLabel:
    """A human confirms or flips a machine proposal, promoting it to ground truth.

    This is the step that makes the label countable. Re-stamping the version
    matters too: the human judged the *current* wording, whatever the machine saw.
    """
    label = session.get(PairLabel, label_id)
    if label is None:
        raise LookupError(f"label {label_id} does not exist")

    concept = session.get(Concept, label.concept_id)
    if concept is not None:
        label.ontology_version = snapshots.resolve_current(session, concept.ontology_id).version

    label.matched = matched
    label.source = ADJUDICATED
    if note:
        label.note = note
    session.flush()
    return label


def record_observation(
    session: Session,
    *,
    concept_id: int,
    locus_id: int,
    occurred_on: _date,
    description: str | None = None,
    source_note: str | None = None,
) -> Observation:
    """Assert that a concept occurred somewhere on a date.

    Positive only, and independent of any run: this is the recall denominator
    that stays valid when the pipeline changes.
    """
    existing = session.scalar(
        select(Observation).where(
            Observation.concept_id == concept_id,
            Observation.locus_id == locus_id,
            Observation.occurred_on == occurred_on,
        )
    )
    if existing is not None:
        if description:
            existing.description = description
        session.flush()
        return existing

    observation = Observation(
        concept_id=concept_id,
        locus_id=locus_id,
        occurred_on=occurred_on,
        description=description,
        source_note=source_note,
    )
    session.add(observation)
    session.flush()
    return observation


def supporting_documents(session: Session, observation_id: int) -> list[Document]:
    """An observation's supporting documents, *derived* from positive pair labels.

    Derived rather than stored: that is what keeps the two tiers independent and
    lets a rejected match exist without an event to hang from.
    """
    observation = session.get(Observation, observation_id)
    if observation is None:
        return []

    query = (
        select(Document)
        .join(PairLabel, PairLabel.document_id == Document.id)
        .where(
            PairLabel.concept_id == observation.concept_id,
            PairLabel.matched.is_(True),
            PairLabel.source.in_(TRUSTED_SOURCES),
        )
    )
    if observation.locus_id is not None:
        query = query.where(PairLabel.locus_id == observation.locus_id)
    if observation.occurred_on is not None:
        query = query.where(PairLabel.occurred_on == observation.occurred_on)
    return list(session.scalars(query))
