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
import hashlib
import io
from dataclasses import dataclass, field
from datetime import date as _date

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.models import (
    TRUSTED_SOURCES,
    Concept,
    Document,
    Locus,
    Observation,
    PairLabel,
)
from hontology.evalkit import labels as label_service
from hontology.ontology import hierarchy

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


def _parse_date(raw: str) -> _date | None:
    value = (raw or "").strip()
    return _date.fromisoformat(value) if value else None


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


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
    concepts = {
        c.id: c.name
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    loci = {locus.id: locus.iso3 for locus in session.scalars(select(Locus))}

    rows = session.scalars(
        select(PairLabel)
        .join(Concept, Concept.id == PairLabel.concept_id)
        .where(Concept.ontology_id == ontology_id)
        .order_by(PairLabel.id)
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LABEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for label in rows:
        document = session.get(Document, label.document_id)
        if document is None:
            continue
        writer.writerow(
            {
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
        )
    return buffer.getvalue()


def export_observations(session: Session, ontology_id: int) -> str:
    """Known occurrences, as CSV. Positive by construction."""
    concepts = {
        c.id: c.name
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    loci = {locus.id: locus.iso3 for locus in session.scalars(select(Locus))}

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=OBSERVATION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for observation in session.scalars(
        select(Observation)
        .join(Concept, Concept.id == Observation.concept_id)
        .where(Concept.ontology_id == ontology_id)
        .order_by(Observation.id)
    ):
        writer.writerow(
            {
                "concept": concepts.get(observation.concept_id, ""),
                "locus_iso3": loci.get(observation.locus_id, ""),
                "occurred_on": observation.occurred_on.isoformat(),
                "description": observation.description or "",
                "source_note": observation.source_note or "",
            }
        )
    return buffer.getvalue()


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
    concepts = {
        c.name: c.id
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    loci = {
        locus.iso3: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso3.is_not(None)))
    }
    report = ImportReport()

    reader = csv.DictReader(io.StringIO(csv_text))
    for line_number, row in enumerate(reader, start=2):
        url = (row.get("document_url") or "").strip()
        concept_name = (row.get("concept") or "").strip()
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
            document = Document(url=url, url_hash=_url_hash(url))
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
        label.source = (row.get("source") or "").strip() or default_source
        label.locus_id = loci.get((row.get("locus_iso3") or "").strip())
        label.occurred_on = occurred_on
        label.note = (row.get("note") or "").strip() or None
        label.proposed_by = (row.get("proposed_by") or "").strip() or None
        # Preserved, not re-stamped: this records which wording was judged, and
        # overwriting it would falsely claim the label saw today's definition.
        label.ontology_version = (row.get("ontology_version") or "").strip() or None

    session.flush()
    return report.as_dict()


def import_observations(session: Session, ontology_id: int, csv_text: str) -> dict:
    """Load known occurrences from CSV."""
    concepts = {
        c.name: c.id
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    loci = {
        locus.iso3: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso3.is_not(None)))
    }
    report = ImportReport()

    reader = csv.DictReader(io.StringIO(csv_text))
    for line_number, row in enumerate(reader, start=2):
        concept_name = (row.get("concept") or "").strip()
        concept_id = concepts.get(concept_name)
        if concept_id is None:
            report.unknown_concepts.append(concept_name or "(blank)")
            continue

        locus_id = loci.get((row.get("locus_iso3") or "").strip())
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
            description=(row.get("description") or "").strip() or None,
            source_note=(row.get("source_note") or "").strip() or None,
        )
        if before is None:
            report.created += 1
        else:
            report.updated += 1

    session.flush()
    return report.as_dict()


# ---------------------------------------------------------------------------
# Whole-document labels
# ---------------------------------------------------------------------------

DOCUMENT_LABEL_COLUMNS = ("position", "document_url", "title", "excerpt", "concepts", "note")
NO_CONCEPT = "none"


def label_document(
    session: Session,
    ontology_id: int,
    document_id: int,
    chosen: set[int],
    *,
    note: str | None = None,
    universe: set[int] | None = None,
) -> dict:
    """Label one whole document: *chosen* positive, every other leaf negative.

    *universe* defaults to the ontology's leaves. Written as ``human`` labels
    stamped with the current version, since a person read the current wording;
    labelling the document again replaces its labels.
    """
    leaves = universe if universe is not None else hierarchy.leaves(session, ontology_id)
    outside = sorted(set(chosen) - leaves)
    if outside:
        raise ValueError(f"not leaves of ontology {ontology_id}: {outside}")
    for concept_id in sorted(leaves):
        label_service.upsert_label(
            session,
            document_id=document_id,
            concept_id=concept_id,
            matched=concept_id in chosen,
            source=label_service.HUMAN,
            note=note,
            ontology_id=ontology_id,
        )
    return {"positives": len(chosen), "negatives": len(leaves) - len(chosen)}


def document_label_status(
    session: Session, ontology_id: int, document_ids: list[int]
) -> dict[int, dict]:
    """Per document: whether every leaf has a counted label, its positives, its note.

    Only trusted labels count, so a machine proposal never marks a document as
    labelled.
    """
    leaves = hierarchy.leaves(session, ontology_id)
    covered: dict[int, set[int]] = {d: set() for d in document_ids}
    positives: dict[int, list[int]] = {d: [] for d in document_ids}
    notes: dict[int, str | None] = dict.fromkeys(document_ids)
    rows = session.execute(
        select(PairLabel.document_id, PairLabel.concept_id, PairLabel.matched, PairLabel.note)
        .where(
            among(PairLabel.document_id, document_ids),
            among(PairLabel.concept_id, leaves),
            PairLabel.source.in_(TRUSTED_SOURCES),
        )
        .order_by(PairLabel.document_id, PairLabel.concept_id)
    )
    for document_id, concept_id, matched, note in rows:
        covered[document_id].add(concept_id)
        if matched:
            positives[document_id].append(concept_id)
        notes[document_id] = notes[document_id] or note
    return {
        d: {
            "labelled": bool(leaves) and covered[d] >= leaves,
            "positives": positives[d],
            "note": notes[d],
        }
        for d in document_ids
    }


def parse_document_labels(
    session: Session,
    ontology_id: int,
    csv_text: str,
    *,
    concept_names: set[str] | None = None,
) -> dict:
    """Read whole-document label rows without writing them.

    Each labelled row becomes its document's id, the ids of the concepts it lists,
    and its note; ``universe`` is every concept a row answers (default: the
    ontology's leaves), so the ones a row does not list are its negatives. Blank
    rows are not yet labelled and are counted, not read. A row naming an unknown
    concept or document is reported in ``errors`` and left out whole.
    """
    known = {
        c.name: c.id
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    # Leaves only: an internal class's answer is derived from its leaves, so a
    # person never labels one, and naming one is an error.
    leaf_ids = hierarchy.leaves(session, ontology_id)
    names = (
        concept_names
        if concept_names is not None
        else {name for name, cid in known.items() if cid in leaf_ids}
    )
    rows: list[dict] = []
    skipped_blank = 0
    errors: list[str] = []
    for line_number, row in enumerate(csv.DictReader(io.StringIO(csv_text)), start=2):
        url = (row.get("document_url") or "").strip()
        listed = (row.get("concepts") or "").strip()
        if not url:
            errors.append(f"line {line_number}: missing document_url")
            continue
        if not listed:
            skipped_blank += 1
            continue
        chosen = (
            set()
            if listed.lower() == NO_CONCEPT
            else {name.strip() for name in listed.split(";") if name.strip()}
        )
        unknown = sorted(chosen - names)
        if unknown:
            errors.append(f"line {line_number}: unknown concepts {unknown}")
            continue
        document = session.scalar(select(Document).where(Document.url == url))
        if document is None:
            errors.append(f"line {line_number}: unknown document {url}")
            continue
        rows.append(
            {
                "document_id": document.id,
                "positives": {known[name] for name in chosen},
                "note": (row.get("note") or "").strip() or None,
            }
        )
    return {
        "rows": rows,
        "universe": {known[name] for name in names},
        "skipped_blank": skipped_blank,
        "errors": errors,
    }


def document_label_map(session: Session, ontology_id: int, csv_text: str) -> dict:
    """A whole-document labels file as a ``(document, concept) -> matched`` map,
    in place of the label bank; refuses a file with any unreadable row."""
    parsed = parse_document_labels(session, ontology_id, csv_text)
    if parsed["errors"]:
        raise ValueError("; ".join(parsed["errors"][:5]))
    return {
        (row["document_id"], concept_id): concept_id in row["positives"]
        for row in parsed["rows"]
        for concept_id in parsed["universe"]
    }


def import_document_labels(
    session: Session,
    ontology_id: int,
    csv_text: str,
    *,
    concept_names: set[str] | None = None,
) -> dict:
    """Load whole-document labels: the concepts that apply, all others negative.

    One row per document. ``concepts`` lists the names that apply separated by
    ``;``; the word ``none`` means none does; blank means not labelled yet and the
    row is skipped. Every concept in *concept_names* (default: the ontology's leaves)
    that the row does not list is written as a negative, which is what makes
    recall measurable: a pair the system never surfaced still has a label.

    Written as ``human`` labels stamped with the current version, since a person
    read the current wording. An unknown name rejects the whole row rather than
    silently labelling the rest of it.
    """
    parsed = parse_document_labels(session, ontology_id, csv_text, concept_names=concept_names)
    documents = positives = negatives = 0
    for row in parsed["rows"]:
        label_document(
            session,
            ontology_id,
            row["document_id"],
            row["positives"],
            note=row["note"],
            universe=parsed["universe"],
        )
        documents += 1
        positives += len(row["positives"])
        negatives += len(parsed["universe"]) - len(row["positives"])
    skipped_blank, errors = parsed["skipped_blank"], parsed["errors"]

    session.flush()
    return {
        "documents": documents,
        "positives": positives,
        "negatives": negatives,
        "skipped_blank": skipped_blank,
        "errors": errors,
    }
