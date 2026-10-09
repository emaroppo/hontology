"""An ontology's structure, for display."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from hontology.apps.api.routers.ontology.lookups import ontology_or_404
from hontology.db.models.ontology import PRECURSOR_OF
from hontology.db.session import get_db
from hontology.ontology import hierarchy, service

router = APIRouter(prefix="/ontologies", tags=["ontology"])


@router.get("/{ontology_id}/hierarchy")
def get_hierarchy(ontology_id: int, db: Session = Depends(get_db)):
    """Every class with its place in the hierarchy, for display.

    Read-only on purpose: structure is authored in an OWL editor and arrives by
    import, so there is nothing here to edit it with.
    """
    ontology_or_404(db, ontology_id)
    concepts = service.list_concepts(db, ontology_id)
    names = {c.id: c.name for c in concepts}
    categories = {c.id: c.name for c in service.list_categories(db, ontology_id)}
    parent_map = hierarchy.parents(db, ontology_id)
    child_map = hierarchy.children(db, ontology_id)
    precursors: dict[int, list[int]] = {}
    for subject, obj in hierarchy.edges(db, ontology_id, PRECURSOR_OF):
        precursors.setdefault(subject, []).append(obj)
    groups: dict[int, list[str]] = {}
    for group in service.list_groups(db, ontology_id):
        for member in group.members:
            groups.setdefault(member.concept_id, []).append(group.name)

    def by_name(ids) -> list[str]:
        return sorted(names[i] for i in ids)

    return {
        "structured": bool(parent_map),
        "classes": [
            {
                "id": c.id,
                "name": c.name,
                "definition": c.definition,
                "inclusion_criteria": c.inclusion_criteria,
                "exclusion_criteria": c.exclusion_criteria,
                "category": categories.get(c.category_id) if c.category_id else None,
                "weight": c.weight,
                "leaf": c.id not in child_map,
                "parents": by_name(parent_map.get(c.id, ())),
                "children": by_name(child_map.get(c.id, ())),
                "precursor_of": by_name(precursors.get(c.id, ())),
                "groups": sorted(groups.get(c.id, ())),
            }
            for c in concepts
        ],
    }
