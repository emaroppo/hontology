"""Machine annotation sets: labels from an annotator that is not a person.

Every score is computed against a *truth*, and there are two kinds:

- **Human**: the label bank, trusted and current labels only (see
  `evaluation.labels.bank.trusted_labels`).
- **Machine**: a named annotation set, such as a blind LLM labelling a sample
  on its own. Scoring against it measures agreement with that annotator, which
  is worth knowing and is not the same as correctness, so it is never mixed
  into the bank.

A set is imported from a whole-document labels file (`document_url,concepts`,
concepts separated by ``;`` or ``none``), read as the label sheet is: listed
leaves are positive and the other leaves negative. Re-importing a set under the
same name replaces its answers for the articles the file covers and keeps the
rest, so a set can grow as its annotator labels further.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import AnnotationSet, MachineAnnotation
from hontology.evaluation.labels.bank import trusted_labels
from hontology.evaluation.labels.document_labels import parse_document_labels
from hontology.ontology import service, snapshots

Truth = dict[tuple[int, int], bool]

HUMAN = None  # the truth that is the label bank


def import_set(
    session: Session,
    ontology_id: int,
    name: str,
    csv_text: str,
    *,
    description: str | None = None,
) -> dict:
    """Create or extend the set *name* from a whole-document labels file.

    The file is read whole first; one unreadable row refuses the import, so a set
    is never left half written.
    """
    service.get_ontology(session, ontology_id)
    parsed = parse_document_labels(session, ontology_id, csv_text)
    if parsed["errors"]:
        raise ValueError("; ".join(parsed["errors"][:5]))
    found = session.scalar(
        select(AnnotationSet).where(
            AnnotationSet.ontology_id == ontology_id, AnnotationSet.name == name
        )
    )
    if found is None:
        found = AnnotationSet(ontology_id=ontology_id, name=name, description=description)
        session.add(found)
        session.flush()
    elif description:
        found.description = description
    version = snapshots.resolve_current(session, ontology_id).version

    existing = {
        (a.document_id, a.concept_id): a
        for a in session.scalars(
            select(MachineAnnotation).where(
                MachineAnnotation.set_id == found.id,
                MachineAnnotation.document_id.in_([r["document_id"] for r in parsed["rows"]]),
            )
        )
    }
    written = 0
    for row in parsed["rows"]:
        for concept_id in parsed["universe"]:
            matched = concept_id in row["positives"]
            note = row["note"] if matched else None
            annotation = existing.get((row["document_id"], concept_id))
            if annotation is None:
                session.add(
                    MachineAnnotation(
                        set_id=found.id,
                        document_id=row["document_id"],
                        concept_id=concept_id,
                        matched=matched,
                        note=note,
                        ontology_version=version,
                    )
                )
            else:
                annotation.matched, annotation.note = matched, note
                annotation.ontology_version = version
            written += 1
    session.flush()
    return {
        "set": name,
        "articles": len(parsed["rows"]),
        "pairs": written,
        "positives": sum(len(r["positives"]) for r in parsed["rows"]),
        "skipped_blank": parsed["skipped_blank"],
    }


def list_sets(session: Session, ontology_id: int) -> list[dict]:
    rows = session.execute(
        select(
            AnnotationSet.name,
            AnnotationSet.description,
            func.count(func.distinct(MachineAnnotation.document_id)),
            func.count(MachineAnnotation.id),
            func.count(MachineAnnotation.id).filter(MachineAnnotation.matched.is_(True)),
        )
        .outerjoin(MachineAnnotation, MachineAnnotation.set_id == AnnotationSet.id)
        .where(AnnotationSet.ontology_id == ontology_id)
        .group_by(AnnotationSet.id)
        .order_by(AnnotationSet.name)
    )
    return [
        {
            "name": name,
            "description": description,
            "articles": articles,
            "pairs": pairs,
            "positives": positives,
        }
        for name, description, articles, pairs, positives in rows
    ]


def human_summary(session: Session, ontology_id: int) -> dict:
    labels = trusted_labels(session, ontology_id)
    return {
        "articles": len({label.document_id for label in labels}),
        "pairs": len(labels),
        "positives": sum(1 for label in labels if label.matched),
    }


def truth(session: Session, ontology_id: int, annotator: str | None = HUMAN) -> Truth:
    """``(document, concept) -> matched`` under the chosen truth: the human
    label bank (trusted, current), or the machine annotation set *annotator*."""
    if annotator is HUMAN:
        return {
            (label.document_id, label.concept_id): bool(label.matched)
            for label in trusted_labels(session, ontology_id)
        }
    found = session.scalar(
        select(AnnotationSet).where(
            AnnotationSet.ontology_id == ontology_id, AnnotationSet.name == annotator
        )
    )
    if found is None:
        raise LookupError(f"no machine annotation set {annotator!r}")
    return {
        (document_id, concept_id): matched
        for document_id, concept_id, matched in session.execute(
            select(
                MachineAnnotation.document_id,
                MachineAnnotation.concept_id,
                MachineAnnotation.matched,
            ).where(MachineAnnotation.set_id == found.id)
        )
    }
