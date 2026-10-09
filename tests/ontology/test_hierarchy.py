"""The class hierarchy: relations, versions, import/export, OWL, and retrieval.

The tests that matter most guard the flat baseline: adding structure to an
ontology must not change a flat ontology's version, the wording its labels
answered, or what a flat run retrieves.
"""

from __future__ import annotations

import hashlib

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS
from sqlalchemy import select

from hontology.db.models import Candidate, Document, Run
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank
from hontology.ontology import hierarchy, owl, service, snapshots

pytestmark = pytest.mark.requires_db

LEAVES = {
    "Import ban": "A government stops imports of a good.",
    "Export ban": "A government stops exports of a good.",
    "Embargo on a country": "A government stops all trade with a country.",
    "Port strike": "Dockworkers stop work.",
}
INTERNAL = {
    "Trade restriction": "Any of: an import ban or an export ban.",
    "Sanctions": "Any of: an embargo, or an import or export ban adopted as a sanction.",
    "Trade policy": "Any trade restriction or sanction.",
}
STRUCTURE = [
    ["Import ban", "subclass_of", "Trade restriction"],
    ["Import ban", "subclass_of", "Sanctions"],
    ["Export ban", "subclass_of", "Trade restriction"],
    ["Export ban", "subclass_of", "Sanctions"],
    ["Embargo on a country", "subclass_of", "Sanctions"],
    ["Trade restriction", "subclass_of", "Trade policy"],
    ["Sanctions", "subclass_of", "Trade policy"],
    ["Embargo on a country", "precursor_of", "Export ban"],
]


def flat_payload(slug: str = "test-hier") -> dict:
    return {
        "export_version": 1,
        "slug": slug,
        "name": "Hierarchy",
        "concepts": [{"name": n, "definition": d} for n, d in LEAVES.items()],
    }


def hierarchical_payload(slug: str = "test-hier") -> dict:
    payload = flat_payload(slug) | {"export_version": 2, "relations": STRUCTURE}
    payload["concepts"] = payload["concepts"] + [
        {"name": n, "definition": d} for n, d in INTERNAL.items()
    ]
    return payload


def ids_by_name(session, ontology_id: int) -> dict[str, int]:
    return {c.name: c.id for c in service.list_concepts(session, ontology_id)}


class TestRelations:
    def test_leaves_and_top_level_follow_the_edges(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            ids = ids_by_name(session, ontology.id)
            assert hierarchy.leaves(session, ontology.id) == {ids[n] for n in LEAVES}
            # A leaf with no parent is top level too: it is still asked about.
            assert hierarchy.top_level(session, ontology.id) == {
                ids["Trade policy"],
                ids["Port strike"],
            }

    def test_a_class_may_have_several_parents(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            ids = ids_by_name(session, ontology.id)
            parents = hierarchy.parents(session, ontology.id)
            assert parents[ids["Import ban"]] == {ids["Trade restriction"], ids["Sanctions"]}
            assert hierarchy.ancestors(parents, ids["Import ban"]) == {
                ids["Trade restriction"],
                ids["Sanctions"],
                ids["Trade policy"],
            }
            assert hierarchy.depth(parents, ids["Import ban"]) == 2

    def test_a_cycle_is_refused_and_the_old_relations_survive(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            before = hierarchy.edges(session, ontology.id)
            looped = hierarchical_payload()
            looped["relations"] = STRUCTURE + [["Trade policy", "subclass_of", "Import ban"]]
            with pytest.raises(service.Conflict, match="cycle"):
                service.import_ontology(session, looped)
            assert hierarchy.edges(session, ontology.id) == before

    def test_a_relation_to_an_unknown_class_is_refused(self):
        bad = hierarchical_payload()
        bad["relations"] = [["Import ban", "subclass_of", "Nowhere"]]
        with session_scope() as session, pytest.raises(service.Conflict, match="Nowhere"):
            service.import_ontology(session, bad)


class TestVersions:
    def test_a_flat_ontology_hashes_as_it_always_did(self):
        """Adding the capability must not mint versions for anyone not using it."""
        with session_scope() as session:
            ontology = service.import_ontology(session, flat_payload())
            concepts = service.list_concepts(session, ontology.id)
            assert snapshots.content_hash(concepts) == snapshots.content_hash(concepts, [])

    def test_structure_changes_the_version(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            first = snapshots.resolve_current(session, ontology.id)
            ids = ids_by_name(session, ontology.id)
            hierarchy.set_relations(
                session,
                ontology.id,
                [(ids["Import ban"], "subclass_of", ids["Trade restriction"])],
            )
            second = snapshots.resolve_current(session, ontology.id)
            assert second.created and second.version != first.version

    def test_leaf_labels_stay_valid_when_a_hierarchy_is_added(self):
        """The internal classes are new and the leaves' wording is untouched, so
        nothing a person labelled has been reworded."""
        with session_scope() as session:
            ontology = service.import_ontology(session, flat_payload())
            ids = ids_by_name(session, ontology.id)
            document = Document(url="https://hi.test/1", url_hash="hitest000001")
            session.add(document)
            session.flush()
            label = bank.upsert_label(
                session,
                document_id=document.id,
                concept_id=ids["Import ban"],
                matched=True,
                ontology_id=ontology.id,
            )
            stamped = label.ontology_version
            service.import_ontology(session, hierarchical_payload(), allow_text_change=False)
            assert snapshots.resolve_current(session, ontology.id).version != stamped
            assert bank.stale_label_ids(session, ontology.id) == set()


class TestImportExport:
    def test_a_version_1_import_leaves_relations_alone(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            before = hierarchy.edges(session, ontology.id)
            service.import_ontology(session, flat_payload())
            assert hierarchy.edges(session, ontology.id) == before

    def test_export_carries_the_relations(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            exported = service.export_ontology(session, ontology.id)
            assert exported["export_version"] == 2
            assert exported["relations"] == sorted(STRUCTURE)

    def test_structure_cannot_silently_reword_a_class(self):
        reworded = hierarchical_payload()
        reworded["concepts"][0]["definition"] = "Something else entirely."
        with session_scope() as session:
            service.import_ontology(session, flat_payload())
            with pytest.raises(service.Conflict, match="reword"):
                service.import_ontology(session, reworded, allow_text_change=False)


class TestOwl:
    def test_two_parents_are_a_union_not_an_intersection(self):
        """Two plain subClassOf statements would mean A *and* B; a second parent
        here means either, which OWL spells owl:unionOf."""
        graph = owl.to_graph(hierarchical_payload() | {"description": None})
        assert len(list(graph.objects(None, OWL.unionOf))) == 2  # Import and Export ban
        base = owl.ONTOLOGY_BASE + "test-hier#"
        restriction = URIRef(base + "Trade_restriction")
        assert list(graph.objects(restriction, RDFS.subClassOf)) == [
            URIRef(base + "Trade_policy")
        ]

    def test_the_round_trip_is_lossless(self):
        payload = hierarchical_payload() | {"description": "A test."}
        back = owl.from_graph(
            Graph().parse(data=owl.to_graph(payload).serialize(), format="turtle")
        )
        assert back["slug"] == payload["slug"]
        assert back["relations"] == sorted(STRUCTURE)
        assert {c["name"]: c["definition"] for c in back["concepts"]} == LEAVES | INTERNAL

    def test_reimporting_its_own_export_changes_nothing(self):
        with session_scope() as session:
            ontology = service.import_ontology(session, hierarchical_payload())
            before = snapshots.resolve_current(session, ontology.id)
            turtle = owl.export_turtle(session, ontology.id)
            owl.import_turtle(session, turtle)
            after = snapshots.resolve_current(session, ontology.id)
            assert (after.version, after.created) == (before.version, False)


class FakeEmbedder:
    """Deterministic three-dimensional vectors, so retrieval runs without a model."""

    name = "fake"

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([digest[0] / 255 + 0.01, digest[1] / 255 + 0.01, digest[2] / 255 + 0.01])
        return out

    def dimension(self, model: str) -> int:
        return 3


def test_flat_retrieval_only_ever_ranks_leaves(tmp_path, monkeypatch):
    """Internal classes added to an ontology must never become candidates, or a
    flat run would change the moment the ontology gained structure."""
    from hontology.config import get_settings
    from hontology.pipeline.retrieve import candidates, embed

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(embed, "get_provider", lambda name, **kw: FakeEmbedder())
    (tmp_path / "leaf.txt").write_text(
        "Dockworkers walked out and a government banned exports."
    )

    with session_scope() as session:
        ontology = service.import_ontology(session, hierarchical_payload())
        other = service.import_ontology(session, flat_payload("test-hier-other"))
        document = Document(
            url="https://hi.test/leaf", url_hash="hileaf000001", body_path="leaf.txt"
        )
        run = Run(
            name="leaf",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="c",
            judge_key="j_c",
            status="running",
        )
        session.add_all([document, run])
        session.flush()
        # The other ontology shares wording with this one; its vectors must not leak in.
        embed.embed_concepts(session, FakeEmbedder(), "fake-model", other.id)
        candidates.build_semantic(
            session,
            run.id,
            ontology_id=ontology.id,
            documents=[document],
            config={
                "embed_provider": "fake",
                "embed_model": "fake-model",
                "concept_fields": "name+definition",
                "selection": "top-k",
                "top_k": 10,
                "pool_size": 20,
                "source": "semantic",
            },
            embed_body_limit=2000,
        )
        ranked = set(
            session.scalars(select(Candidate.concept_id).where(Candidate.run_id == run.id))
        )
        assert ranked == hierarchy.leaves(session, ontology.id)
