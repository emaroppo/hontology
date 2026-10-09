"""Moving the ground-truth bank in and out as CSV.

The bank is the expensive asset in this project — a few hundred labels is hours
of human attention — so it has to be portable, reviewable in a diff, and safe to
re-import. Five decisions make it so:

**Keyed by URL and concept name, not database ids.** Ids are local to one
database; a bank exported here and imported elsewhere would silently attach to
whatever happened to hold those ids. Names and URLs mean the same thing
everywhere.

**``ontology_version`` is preserved, never re-stamped.** The version records
*which wording the judgment was made against*. Re-stamping an imported label
with today's version would claim a human read the current definition when they
did not, which quietly defeats staleness detection — the one mechanism protecting
these labels from rotting.

**Existing labels are not overwritten by default.** Importing must never destroy
adjudicated work by accident; overwriting is opt-in and reported.

**Unknown documents are created as stubs.** Labels legitimately arrive before the
corpus does — from a colleague, from an earlier corpus, from a spreadsheet. A
stub row carries the URL and no body, and the scraper fills it in later.

**Unknown concepts are skipped and named in the report.** A label for a concept
this ontology does not have cannot be honoured, and inventing the concept would
be worse than refusing.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date as _date

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import (
    concept_ids_by_name,
    concept_names,
    locus_ids_by_iso3,
    locus_iso3s,
)
from hontology.db.models import (
    Concept,
    Document,
    Observation,
    PairLabel,
)
from hontology.evaluation.labels import bank as label_service
from hontology.pipeline.ingest.feed.slices import url_hash

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

TRUTHY = {"1", "true", "t", "yes", "y", "match", "matched"}
FALSY = {"0", "false", "f", "no", "n", "nomatch", "not matched"}


@dataclass
class ImportReport:
    created: int = 0
    updated: int = 0
    skipped_existing: int = 0
    documents_created: int = 0
    unknown_concepts: list[str] = field(default_factory=list)
    bad_rows: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "created": self.created,
            "updated": self.updated,
            "skipped_existing": self.skipped_existing,
            "documents_created": self.documents_created,
            "unknown_concepts": sorted(set(self.unknown_concepts)),
            "bad_rows": self.bad_rows[:20],
            "bad_row_count": len(self.bad_rows),
        }


def _parse_bool(raw: str) -> bool:
    value = (raw or "").strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    raise ValueError(f"cannot read {raw!r} as a true/false value")


def cell(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def to_csv(rows: Iterable[dict], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _parse_date(raw: str) -> _date | None:
    value = (raw or "").strip()
    return _date.fromisoformat(value) if value else None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_labels(session: Session, ontology_id: int) -> str:
    """Every pair label for an ontology, as CSV.

    Stale labels are included and flagged rather than dropped: the flag is
    information the recipient needs, and silently omitting them would make an
    export look smaller than the work that went into it.
    """
    stale = label_service.stale_label_ids(session, ontology_id)
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


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_labels(
    session: Session,
    ontology_id: int,
    csv_text: str,
    *,
    overwrite: bool = False,
    create_missing_documents: bool = True,
    default_source: str = label_service.IMPORTED,
) -> dict:
    """Load labels from CSV, matching on URL and concept name."""
    concepts = concept_ids_by_name(session, ontology_id)
    loci = locus_ids_by_iso3(session)
    report = ImportReport()

    reader = csv.DictReader(io.StringIO(csv_text))
    for line_number, row in enumerate(reader, start=2):
        url = cell(row, "document_url")
        concept_name = cell(row, "concept")
        if not url or not concept_name:
            report.bad_rows.append(f"line {line_number}: missing document_url or concept")
            continue

        concept_id = concepts.get(concept_name)
        if concept_id is None:
            report.unknown_concepts.append(concept_name)
            continue

        try:
            matched = _parse_bool(row.get("matched", ""))
            occurred_on = _parse_date(row.get("occurred_on", ""))
        except ValueError as exc:
            report.bad_rows.append(f"line {line_number}: {exc}")
            continue

        document = session.scalar(select(Document).where(Document.url == url))
        if document is None:
            if not create_missing_documents:
                report.bad_rows.append(f"line {line_number}: unknown document {url}")
                continue
            # A stub: the label is the asset, and the body can be fetched later.
            document = Document(url=url, url_hash=url_hash(url))
            session.add(document)
            session.flush()
            report.documents_created += 1

        existing = session.scalar(
            select(PairLabel).where(
                PairLabel.document_id == document.id, PairLabel.concept_id == concept_id
            )
        )
        if existing is not None and not overwrite:
            report.skipped_existing += 1
            continue

        label = existing or PairLabel(document_id=document.id, concept_id=concept_id)
        if existing is None:
            session.add(label)
            report.created += 1
        else:
            report.updated += 1

        label.matched = matched
        label.source = cell(row, "source") or default_source
        label.locus_id = loci.get(cell(row, "locus_iso3"))
        label.occurred_on = occurred_on
        label.note = cell(row, "note") or None
        label.proposed_by = cell(row, "proposed_by") or None
        # Preserved, not re-stamped: this records which wording was judged, and
        # overwriting it would falsely claim the label saw today's definition.
        label.ontology_version = cell(row, "ontology_version") or None

    session.flush()
    return report.as_dict()


def import_observations(session: Session, ontology_id: int, csv_text: str) -> dict:
    """Load known occurrences from CSV."""
    concepts = concept_ids_by_name(session, ontology_id)
    loci = locus_ids_by_iso3(session)
    report = ImportReport()

    reader = csv.DictReader(io.StringIO(csv_text))
    for line_number, row in enumerate(reader, start=2):
        concept_name = cell(row, "concept")
        concept_id = concepts.get(concept_name)
        if concept_id is None:
            report.unknown_concepts.append(concept_name or "(blank)")
            continue

        locus_id = loci.get(cell(row, "locus_iso3"))
        if locus_id is None:
            report.bad_rows.append(
                f"line {line_number}: unknown locus {row.get('locus_iso3')!r}"
            )
            continue

        try:
            occurred_on = _parse_date(row.get("occurred_on", ""))
        except ValueError as exc:
            report.bad_rows.append(f"line {line_number}: {exc}")
            continue
        if occurred_on is None:
            report.bad_rows.append(f"line {line_number}: occurred_on is required")
            continue

        before = session.scalar(
            select(Observation).where(
                Observation.concept_id == concept_id,
                Observation.locus_id == locus_id,
                Observation.occurred_on == occurred_on,
            )
        )
        label_service.record_observation(
            session,
            concept_id=concept_id,
            locus_id=locus_id,
            occurred_on=occurred_on,
            description=cell(row, "description") or None,
            source_note=cell(row, "source_note") or None,
        )
        if before is None:
            report.created += 1
        else:
            report.updated += 1

    session.flush()
    return report.as_dict()
