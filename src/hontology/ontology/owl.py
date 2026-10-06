"""OWL interchange: the ontology as a standard file Protégé can open and edit.

Postgres stays the source of truth (versions, labels and runs all hang off its
rows); this module only translates. An ontology exports as Turtle and imports
back through the same path as a JSON export, so both formats obey the same
rules: relations replaced as a set, cycles refused, and optionally no rewording
of existing classes.

**Mapping.**

- Each class is an ``owl:Class`` with its name as ``rdfs:label``.
- Its wording is three annotation properties in the ``hontology:`` namespace:
  definition, inclusion criteria and exclusion criteria.
- A single parent is a plain ``rdfs:subClassOf``.
- Several parents are ``rdfs:subClassOf [ owl:unionOf (A B) ]``. That is OWL's
  spelling of "a member of A or of B", the union reading this project gives a
  second parent. Two plain ``rdfs:subClassOf`` statements would mean A *and* B.
- ``precursor_of`` relates classes, not individuals, so it is an annotation
  property rather than an object property.
"""

from __future__ import annotations

import re
from typing import Any

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD
from sqlalchemy.orm import Session

from hontology.db.models import Ontology
from hontology.db.models.ontology import PRECURSOR_OF, SUBCLASS_OF
from hontology.ontology import service

HON = Namespace("https://github.com/emaroppo/hontology/ns#")
ONTOLOGY_BASE = "https://github.com/emaroppo/hontology/ontology/"

_TEXT = {
    "definition": HON.definition,
    "inclusion_criteria": HON.inclusionCriteria,
    "exclusion_criteria": HON.exclusionCriteria,
}


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def to_graph(payload: dict[str, Any]) -> Graph:
    """An exported ontology dict (as `service.export_ontology` returns) as a graph."""
    base = Namespace(f"{ONTOLOGY_BASE}{payload['slug']}#")
    graph = Graph()
    graph.bind("owl", OWL)
    graph.bind("hontology", HON)
    graph.bind(payload["slug"].replace("-", "_"), base)

    header = URIRef(f"{ONTOLOGY_BASE}{payload['slug']}")
    graph.add((header, RDF.type, OWL.Ontology))
    graph.add((header, RDFS.label, Literal(payload["name"])))
    graph.add((header, HON.slug, Literal(payload["slug"])))
    if payload.get("description"):
        graph.add((header, RDFS.comment, Literal(payload["description"])))
    for prop in (*_TEXT.values(), HON.category, HON.weight, HON.precursorOf, HON.slug):
        graph.add((prop, RDF.type, OWL.AnnotationProperty))

    iri = {c["name"]: base[_slug(c["name"])] for c in payload["concepts"]}
    for concept in payload["concepts"]:
        node = iri[concept["name"]]
        graph.add((node, RDF.type, OWL.Class))
        graph.add((node, RDFS.label, Literal(concept["name"])))
        for field, prop in _TEXT.items():
            if concept.get(field):
                graph.add((node, prop, Literal(concept[field])))
        if concept.get("category"):
            graph.add((node, HON.category, Literal(concept["category"])))
        if concept.get("weight") is not None:
            graph.add((node, HON.weight, Literal(concept["weight"], datatype=XSD.double)))

    parents: dict[str, list[str]] = {}
    for subject, predicate, obj in payload.get("relations", []):
        if predicate == SUBCLASS_OF:
            parents.setdefault(subject, []).append(obj)
        elif predicate == PRECURSOR_OF:
            graph.add((iri[subject], HON.precursorOf, iri[obj]))
    for child, above in parents.items():
        if len(above) == 1:
            graph.add((iri[child], RDFS.subClassOf, iri[above[0]]))
            continue
        union = BNode()
        members = BNode()
        graph.add((union, RDF.type, OWL.Class))
        Collection(graph, members, [iri[name] for name in sorted(above)])
        graph.add((union, OWL.unionOf, members))
        graph.add((iri[child], RDFS.subClassOf, union))
    return graph


def from_graph(graph: Graph) -> dict[str, Any]:
    """A graph back into the dict `service.import_ontology` takes."""
    header = next(graph.subjects(RDF.type, OWL.Ontology), None)
    if header is None:
        raise service.Conflict("no owl:Ontology in the file")
    slug = graph.value(header, HON.slug)
    if slug is None:
        raise service.Conflict("the ontology carries no hontology:slug")

    names: dict[Any, str] = {}
    concepts = []
    for node in sorted(graph.subjects(RDF.type, OWL.Class), key=str):
        if isinstance(node, BNode):
            continue  # a union expression, not a class of its own
        label = graph.value(node, RDFS.label)
        if label is None:
            raise service.Conflict(f"class {node} has no rdfs:label")
        names[node] = str(label)
        weight = graph.value(node, HON.weight)
        category = graph.value(node, HON.category)
        concepts.append(
            {
                "name": str(label),
                **{field: _text(graph.value(node, prop)) for field, prop in _TEXT.items()},
                "category": str(category) if category is not None else None,
                "weight": float(str(weight)) if weight is not None else None,
            }
        )

    relations: list[list[str]] = []
    for child, target in graph.subject_objects(RDFS.subClassOf):
        if child not in names:
            continue
        members = graph.value(target, OWL.unionOf) if isinstance(target, BNode) else None
        parents = list(Collection(graph, members)) if members is not None else [target]
        for parent in parents:
            if parent not in names:
                raise service.Conflict(f"{names[child]!r} has a parent that is not a class")
            relations.append([names[child], SUBCLASS_OF, names[parent]])
    for subject, obj in graph.subject_objects(HON.precursorOf):
        relations.append([names[subject], PRECURSOR_OF, names[obj]])

    description = graph.value(header, RDFS.comment)
    return {
        "export_version": service.EXPORT_VERSION,
        "slug": str(slug),
        "name": str(graph.value(header, RDFS.label) or slug),
        "description": str(description) if description is not None else None,
        "concepts": concepts,
        "relations": sorted(relations),
    }


def _text(value: Any) -> str | None:
    return str(value) if value is not None else None


def export_turtle(session: Session, ontology_id: int) -> str:
    return to_graph(service.export_ontology(session, ontology_id)).serialize(format="turtle")


def import_turtle(session: Session, text: str, *, allow_text_change: bool = False) -> Ontology:
    """Import a Turtle file. Rewording existing classes is refused by default,
    since this is the path a hierarchy arrives by and labels depend on wording."""
    payload = from_graph(Graph().parse(data=text, format="turtle"))
    return service.import_ontology(session, payload, allow_text_change=allow_text_change)
