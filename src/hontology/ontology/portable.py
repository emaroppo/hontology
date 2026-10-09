"""Ontology import and export, as a portable dict.

Import is an upsert keyed by name within an ontology, which makes re-importing an
edited export a merge rather than a duplication. That matters because the export
is the intended way to move an ontology between machines and to keep it in
version control alongside the labels that reference it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import ConceptGroup, ConceptRelation, Ontology
from hontology.ontology import hierarchy
from hontology.ontology.service import (
    Conflict,
    create_concept,
    create_ontology,
    get_ontology,
    get_ontology_by_slug,
    get_or_create_category,
    list_categories,
    list_concepts,
    list_groups,
    set_group_members,
)

EXPORT_VERSION = 2
# Version 1 carried no relations; it still imports, leaving relations untouched.
SUPPORTED_EXPORT_VERSIONS = (1, 2)
TEXT_FIELDS = ("definition", "inclusion_criteria", "exclusion_criteria")


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

    _import_concepts(session, ontology, payload.get("concepts", []), allow_text_change)
    session.flush()
    _import_groups(session, ontology, payload.get("groups", []))
    if "relations" in payload:
        _import_relations(session, ontology, payload["relations"])
    return ontology


def _import_concepts(
    session: Session, ontology: Ontology, rows: list[dict[str, Any]], allow_text_change: bool
) -> None:
    """Upsert concepts by name, refusing a rewording unless it is allowed."""
    by_name = {c.name: c for c in list_concepts(session, ontology.id)}
    for row in rows:
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
