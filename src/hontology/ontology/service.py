"""Ontology CRUD: ontologies, categories, concepts and groups.

Every mutation that touches concept *wording* leaves the snapshot layer to notice
— nothing here mints a version. Versions are resolved lazily when a run starts or
a label is written, so an editing session produces one version rather than one
per save. Import and export live in `ontology.portable`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from hontology.db.models import (
    Category,
    Concept,
    ConceptGroup,
    ConceptGroupMember,
    Ontology,
)


class NotFound(LookupError):
    """A referenced row does not exist. The API maps this to a 404."""


class Conflict(ValueError):
    """A uniqueness rule would be violated. The API maps this to a 409."""


# ---------------------------------------------------------------------------
# Ontologies
# ---------------------------------------------------------------------------


def list_ontologies(session: Session) -> list[Ontology]:
    return list(session.scalars(select(Ontology).order_by(Ontology.name)))


def get_ontology(session: Session, ontology_id: int) -> Ontology:
    ontology = session.get(Ontology, ontology_id)
    if ontology is None:
        raise NotFound(f"ontology {ontology_id} does not exist")
    return ontology


def get_ontology_by_slug(session: Session, slug: str) -> Ontology | None:
    return session.scalar(select(Ontology).where(Ontology.slug == slug))


def create_ontology(
    session: Session, *, slug: str, name: str, description: str | None = None
) -> Ontology:
    if get_ontology_by_slug(session, slug) is not None:
        raise Conflict(f"an ontology with slug {slug!r} already exists")
    ontology = Ontology(slug=slug, name=name, description=description)
    session.add(ontology)
    session.flush()
    return ontology


def delete_ontology(session: Session, ontology_id: int) -> None:
    session.delete(get_ontology(session, ontology_id))


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


def get_or_create_category(session: Session, ontology_id: int, name: str) -> Category:
    name = name.strip()
    existing = session.scalar(
        select(Category).where(Category.ontology_id == ontology_id, Category.name == name)
    )
    if existing is not None:
        return existing
    category = Category(ontology_id=ontology_id, name=name)
    session.add(category)
    session.flush()
    return category


def list_categories(session: Session, ontology_id: int) -> list[Category]:
    return list(
        session.scalars(
            select(Category).where(Category.ontology_id == ontology_id).order_by(Category.name)
        )
    )


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


def list_concepts(session: Session, ontology_id: int) -> list[Concept]:
    return list(
        session.scalars(
            select(Concept).where(Concept.ontology_id == ontology_id).order_by(Concept.name)
        )
    )


def get_concept(session: Session, concept_id: int) -> Concept:
    concept = session.get(Concept, concept_id)
    if concept is None:
        raise NotFound(f"concept {concept_id} does not exist")
    return concept


def create_concept(
    session: Session,
    ontology_id: int,
    *,
    name: str,
    definition: str | None = None,
    inclusion_criteria: str | None = None,
    exclusion_criteria: str | None = None,
    category: str | None = None,
    weight: float | None = None,
) -> Concept:
    get_ontology(session, ontology_id)
    name = name.strip()
    if not name:
        raise Conflict("concept name is required")

    duplicate = session.scalar(
        select(Concept).where(Concept.ontology_id == ontology_id, Concept.name == name)
    )
    if duplicate is not None:
        raise Conflict(f"concept {name!r} already exists in this ontology")

    concept = Concept(
        ontology_id=ontology_id,
        name=name,
        definition=definition,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        weight=weight,
        category_id=(
            get_or_create_category(session, ontology_id, category).id if category else None
        ),
    )
    session.add(concept)
    session.flush()
    return concept


def update_concept(session: Session, concept_id: int, **fields: Any) -> Concept:
    concept = get_concept(session, concept_id)

    if "category" in fields:
        category = fields.pop("category")
        concept.category_id = (
            get_or_create_category(session, concept.ontology_id, category).id
            if category
            else None
        )

    for key, value in fields.items():
        if value is None:
            continue
        if not hasattr(concept, key):
            raise Conflict(f"unknown concept field {key!r}")
        setattr(concept, key, value.strip() if isinstance(value, str) else value)

    session.flush()
    return concept


def delete_concept(session: Session, concept_id: int) -> None:
    session.delete(get_concept(session, concept_id))


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def list_groups(session: Session, ontology_id: int) -> list[ConceptGroup]:
    return list(
        session.scalars(
            select(ConceptGroup)
            .where(ConceptGroup.ontology_id == ontology_id)
            .order_by(ConceptGroup.name)
        )
    )


def set_group_members(
    session: Session, group_id: int, members: dict[int, float | None]
) -> None:
    """Replace a group's membership wholesale with ``{concept_id: weight}``."""
    session.execute(delete(ConceptGroupMember).where(ConceptGroupMember.group_id == group_id))
    for concept_id, weight in members.items():
        session.add(ConceptGroupMember(group_id=group_id, concept_id=concept_id, weight=weight))
    session.flush()
