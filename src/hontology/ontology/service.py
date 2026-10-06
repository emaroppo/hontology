"""Ontology CRUD, import and export.

Every mutation that touches concept *wording* leaves the snapshot layer to notice
— nothing here mints a version. Versions are resolved lazily when a run starts or
a label is written, so an editing session produces one version rather than one
per save.

Import is an upsert keyed by name within an ontology, which makes re-importing an
edited export a merge rather than a duplication. That matters because the export
is the intended way to move an ontology between machines and to keep it in
version control alongside the labels that reference it.
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
    ConceptRelation,
    Ontology,
)
from hontology.ontology import hierarchy

EXPORT_VERSION = 2
# Version 1 carried no relations; it still imports, leaving relations untouched.
SUPPORTED_EXPORT_VERSIONS = (1, 2)
TEXT_FIELDS = ("definition", "inclusion_criteria", "exclusion_criteria")


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


def fork_group(session: Session, group_id: int, *, name: str) -> ConceptGroup:
    """Copy a group and its weighted membership into an editable variant.

    The copy keeps ``parent_id`` pointing at the original, and its own edge
    weights, so retuning the variant never disturbs what it was forked from.
    """
    original = session.get(ConceptGroup, group_id)
    if original is None:
        raise NotFound(f"group {group_id} does not exist")

    fork = ConceptGroup(
        ontology_id=original.ontology_id,
        name=name,
        description=original.description,
        parent_id=original.id,
    )
    session.add(fork)
    session.flush()

    for member in original.members:
        session.add(
            ConceptGroupMember(
                group_id=fork.id, concept_id=member.concept_id, weight=member.weight
            )
        )
    session.flush()
    return fork


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------


def export_ontology(session: Session, ontology_id: int) -> dict[str, Any]:
    """Serialize an ontology to a portable dict.

    Concepts are exported by *name*, not by database id, so an import into a
    different database merges by name rather than colliding on ids.
    """
    ontology = get_ontology(session, ontology_id)
    categories = {c.id: c.name for c in list_categories(session, ontology_id)}
    groups = list_groups(session, ontology_id)
    group_names = {g.id: g.name for g in groups}

    return {
        "export_version": EXPORT_VERSION,
        "slug": ontology.slug,
        "name": ontology.name,
        "description": ontology.description,
        "concepts": [
            {
                "name": c.name,
                "definition": c.definition,
                "inclusion_criteria": c.inclusion_criteria,
                "exclusion_criteria": c.exclusion_criteria,
                "category": categories.get(c.category_id) if c.category_id else None,
                "weight": c.weight,
            }
            for c in list_concepts(session, ontology_id)
        ],
        "groups": [
            {
                "name": g.name,
                "description": g.description,
                "parent": group_names.get(g.parent_id) if g.parent_id else None,
                "members": [{"concept": m.concept.name, "weight": m.weight} for m in g.members],
            }
            for g in groups
        ],
        "relations": _export_relations(session, ontology_id),
    }


def _export_relations(session: Session, ontology_id: int) -> list[list[str]]:
    names = {c.id: c.name for c in list_concepts(session, ontology_id)}
    return sorted(
        [names[r.subject_id], r.predicate, names[r.object_id]]
        for r in session.scalars(
            select(ConceptRelation).where(ConceptRelation.ontology_id == ontology_id)
        )
    )


def import_ontology(
    session: Session, payload: dict[str, Any], *, allow_text_change: bool = True
) -> Ontology:
    """Create or merge an ontology from an exported dict.

    Upserts by ``slug`` for the ontology and by concept ``name`` within it, so
    re-importing an edited export updates in place instead of duplicating.

    With *allow_text_change* off, an import may add classes and relations but
    may not reword an existing class: adding structure to an ontology must never
    silently change the questions its labels answered.

    A payload carrying ``relations`` replaces the ontology's relations; one
    without them (every version 1 export) leaves them as they are.
    """
    version = payload.get("export_version", EXPORT_VERSION)
    if version not in SUPPORTED_EXPORT_VERSIONS:
        raise Conflict(f"unsupported export_version {version!r}")

    slug = str(payload["slug"]).strip()
    ontology = get_ontology_by_slug(session, slug)
    if ontology is None:
        ontology = create_ontology(
            session,
            slug=slug,
            name=str(payload.get("name") or slug),
            description=payload.get("description"),
        )
    else:
        ontology.name = str(payload.get("name") or ontology.name)
        ontology.description = payload.get("description", ontology.description)

    by_name = {c.name: c for c in list_concepts(session, ontology.id)}
    for row in payload.get("concepts", []):
        name = str(row["name"]).strip()
        fields = {
            "definition": row.get("definition"),
            "inclusion_criteria": row.get("inclusion_criteria"),
            "exclusion_criteria": row.get("exclusion_criteria"),
            "weight": row.get("weight"),
        }
        if name in by_name:
            concept = by_name[name]
            if not allow_text_change:
                changed = [
                    key
                    for key in TEXT_FIELDS
                    if (getattr(concept, key) or "").strip() != (fields[key] or "").strip()
                ]
                if changed:
                    raise Conflict(
                        f"import would reword {name!r} ({', '.join(changed)}); "
                        "pass allow_text_change to permit it"
                    )
            for key, value in fields.items():
                setattr(concept, key, value)
            concept.category_id = (
                get_or_create_category(session, ontology.id, row["category"]).id
                if row.get("category")
                else None
            )
        else:
            create_concept(
                session,
                ontology.id,
                name=name,
                category=row.get("category"),
                **fields,
            )

    session.flush()
    _import_groups(session, ontology, payload.get("groups", []))
    if "relations" in payload:
        _import_relations(session, ontology, payload["relations"])
    return ontology


def _import_relations(session: Session, ontology: Ontology, rows: list[list[str]]) -> None:
    ids = {c.name: c.id for c in list_concepts(session, ontology.id)}
    unknown = sorted({name for row in rows for name in (row[0], row[2]) if name not in ids})
    if unknown:
        raise Conflict(f"relations name unknown classes: {', '.join(unknown)}")
    try:
        hierarchy.set_relations(
            session, ontology.id, [(ids[s], str(p), ids[o]) for s, p, o in rows]
        )
    except hierarchy.HierarchyError as exc:
        raise Conflict(str(exc)) from exc


def _import_groups(session: Session, ontology: Ontology, rows: list[dict[str, Any]]) -> None:
    concepts = {c.name: c.id for c in list_concepts(session, ontology.id)}
    groups = {g.name: g for g in list_groups(session, ontology.id)}

    # Two passes: create every group before wiring parents, so a fork listed
    # before its parent in the file still resolves.
    for row in rows:
        name = str(row["name"]).strip()
        if name not in groups:
            group = ConceptGroup(
                ontology_id=ontology.id, name=name, description=row.get("description")
            )
            session.add(group)
            session.flush()
            groups[name] = group

    for row in rows:
        group = groups[str(row["name"]).strip()]
        parent_name = row.get("parent")
        group.parent_id = groups[parent_name].id if parent_name in groups else None
        set_group_members(
            session,
            group.id,
            {
                concepts[m["concept"]]: m.get("weight")
                for m in row.get("members", [])
                if m.get("concept") in concepts
            },
        )
