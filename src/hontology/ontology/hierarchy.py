"""The class hierarchy: who is under whom, and which classes are leaves.

`subclass_of` edges make an ontology a directed acyclic graph. A class may have
several parents, read as *either*: an import ban sits under both Trade
restriction and Sanctions because it can be adopted as either, not because every
ban is a sanction. The top level is every class with no parent; there is no
separate root, since the top-level question already answers "is this about
anything here at all".

A **leaf** is a class with no children. Flat retrieval and flat judging work on
leaves only, which for a flat ontology is every concept, so adding internal
classes to an ontology never changes what a flat run retrieves or asks.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, ConceptRelation
from hontology.db.models.ontology import PREDICATES, SUBCLASS_OF


class HierarchyError(ValueError):
    pass


def edges(
    session: Session, ontology_id: int, predicate: str = SUBCLASS_OF
) -> list[tuple[int, int]]:
    """``(subject, object)`` pairs of one relation, sorted for stability."""
    return sorted(
        (subject, obj)
        for subject, obj in session.execute(
            select(ConceptRelation.subject_id, ConceptRelation.object_id).where(
                ConceptRelation.ontology_id == ontology_id,
                ConceptRelation.predicate == predicate,
            )
        )
    )


def concept_ids(session: Session, ontology_id: int) -> set[int]:
    return set(session.scalars(select(Concept.id).where(Concept.ontology_id == ontology_id)))


def parents(session: Session, ontology_id: int) -> dict[int, set[int]]:
    """``{class: its parents}``; classes with no parent are absent."""
    out: dict[int, set[int]] = defaultdict(set)
    for child, parent in edges(session, ontology_id):
        out[child].add(parent)
    return dict(out)


def children(session: Session, ontology_id: int) -> dict[int, set[int]]:
    """``{class: its children}``; leaves are absent."""
    out: dict[int, set[int]] = defaultdict(set)
    for child, parent in edges(session, ontology_id):
        out[parent].add(child)
    return dict(out)


def leaves(session: Session, ontology_id: int) -> set[int]:
    """Classes with no children: every concept, in a flat ontology."""
    return concept_ids(session, ontology_id) - set(children(session, ontology_id))


def top_level(session: Session, ontology_id: int) -> set[int]:
    """Classes with no parent: where a hierarchical descent starts."""
    return concept_ids(session, ontology_id) - set(parents(session, ontology_id))


def ancestors(parent_map: dict[int, set[int]], class_id: int) -> set[int]:
    """Every class above *class_id*, through any parent."""
    seen: set[int] = set()
    stack = list(parent_map.get(class_id, ()))
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(parent_map.get(current, ()))
    return seen


def depth(parent_map: dict[int, set[int]], class_id: int) -> int:
    """Length of the longest path up to the top level (top level is 0)."""
    above = parent_map.get(class_id)
    if not above:
        return 0
    return 1 + max(depth(parent_map, parent) for parent in above)


def _creates_cycle(parent_map: dict[int, set[int]], child: int, parent: int) -> bool:
    return parent == child or child in ancestors(parent_map, parent)


def check_acyclic(pairs: list[tuple[int, int]]) -> None:
    """Refuse a subclass_of edge set that loops back on itself."""
    parent_map: dict[int, set[int]] = defaultdict(set)
    for child, parent in pairs:
        if _creates_cycle(parent_map, child, parent):
            raise HierarchyError(f"subclass_of cycle through classes {child} and {parent}")
        parent_map[child].add(parent)


def set_relations(
    session: Session, ontology_id: int, triples: list[tuple[int, str, int]]
) -> None:
    """Replace the ontology's relations with *triples* (subject, predicate, object).

    The whole set is validated first, so a bad file leaves the old relations in
    place: unknown predicates, classes from another ontology, and subclass
    cycles are all refused.
    """
    known = concept_ids(session, ontology_id)
    for subject, predicate, obj in triples:
        if predicate not in PREDICATES:
            raise HierarchyError(f"unknown predicate {predicate!r}")
        if subject not in known or obj not in known:
            raise HierarchyError("relation between classes outside this ontology")
    check_acyclic([(s, o) for s, p, o in triples if p == SUBCLASS_OF])

    for row in session.scalars(
        select(ConceptRelation).where(ConceptRelation.ontology_id == ontology_id)
    ):
        session.delete(row)
    session.flush()
    for subject, predicate, obj in sorted(set(triples)):
        session.add(
            ConceptRelation(
                ontology_id=ontology_id, subject_id=subject, predicate=predicate, object_id=obj
            )
        )
    session.flush()
