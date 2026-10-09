"""Exporting the label bank and known occurrences as CSV.

Keyed by URL and concept name, with each label's ``ontology_version`` as it was
recorded, so an export re-imports anywhere (see `labels.csv_io`).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import concept_names, locus_iso3s
from hontology.db.models import Concept, Document, Observation, PairLabel
from hontology.evaluation.labels import staleness
from hontology.evaluation.labels.csv_format import to_csv

LABEL_COLUMNS = [
    "document_url",
    "concept",
    "matched",
    "source",
    "locus_iso3",
    "occurred_on",
    "note",
    "proposed_by",
    "ontology_version",
    "stale",
]

OBSERVATION_COLUMNS = [
    "concept",
    "locus_iso3",
    "occurred_on",
    "description",
    "source_note",
]


def export_labels(session: Session, ontology_id: int) -> str:
    """Every pair label for an ontology, as CSV.

    Stale labels are included and flagged rather than dropped: the flag is
    information the recipient needs, and silently omitting them would make an
    export look smaller than the work that went into it.
    """
    stale = staleness.stale_label_ids(session, ontology_id)
    concepts = concept_names(session, ontology_id)
    loci = locus_iso3s(session)

    rows = session.scalars(
        select(PairLabel)
        .join(Concept, Concept.id == PairLabel.concept_id)
        .where(Concept.ontology_id == ontology_id)
        .order_by(PairLabel.id)
    )

    def row(label: PairLabel) -> dict | None:
        document = session.get(Document, label.document_id)
        if document is None:
            return None
        return {
            "document_url": document.url,
            "concept": concepts.get(label.concept_id, ""),
            "matched": "true" if label.matched else "false",
            "source": label.source,
            "locus_iso3": loci.get(label.locus_id, "") if label.locus_id else "",
            "occurred_on": label.occurred_on.isoformat() if label.occurred_on else "",
            "note": label.note or "",
            "proposed_by": label.proposed_by or "",
            "ontology_version": label.ontology_version or "",
            "stale": "true" if label.id in stale else "false",
        }

    return to_csv(filter(None, map(row, rows)), LABEL_COLUMNS)


def export_observations(session: Session, ontology_id: int) -> str:
    """Known occurrences, as CSV. Positive by construction."""
    concepts = concept_names(session, ontology_id)
    loci = locus_iso3s(session)

    observations = session.scalars(
        select(Observation)
        .join(Concept, Concept.id == Observation.concept_id)
        .where(Concept.ontology_id == ontology_id)
        .order_by(Observation.id)
    )
    return to_csv(
        (
            {
                "concept": concepts.get(observation.concept_id, ""),
                "locus_iso3": loci.get(observation.locus_id, ""),
                "occurred_on": observation.occurred_on.isoformat(),
                "description": observation.description or "",
                "source_note": observation.source_note or "",
            }
            for observation in observations
        ),
        OBSERVATION_COLUMNS,
    )
