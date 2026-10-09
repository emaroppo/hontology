"""Whole-document labels: a person answers every leaf for one article at once.

One row per document lists the concepts that apply; every other leaf is written
as a negative, which is what makes recall measurable — a pair the system never
surfaced still has a label.
"""

from __future__ import annotations

import csv
import io

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.base import among
from hontology.db.lookups import concept_ids_by_name
from hontology.db.models import TRUSTED_SOURCES, Document, PairLabel
from hontology.evalkit import labels as label_service
from hontology.evalkit.label_io import cell
from hontology.ontology import hierarchy

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
    known = concept_ids_by_name(session, ontology_id)
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
        url = cell(row, "document_url")
        listed = cell(row, "concepts")
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
                "note": cell(row, "note") or None,
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
