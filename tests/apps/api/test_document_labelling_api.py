"""Labelling a sample a whole document at a time, through the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hontology.apps.api.main import app
from hontology.db.models import Document
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank as label_service
from hontology.ontology import service

pytestmark = pytest.mark.requires_db

PAYLOAD = {
    "export_version": 2,
    "slug": "test-document-labelling",
    "name": "Document labelling",
    "concepts": [
        {"name": n, "definition": f"{n}.", "inclusion_criteria": f"{n} counts."}
        for n in ("Trade", "Sanction", "Tariff", "Ban", "Embargo", "Hack")
    ],
    "relations": [
        ["Tariff", "subclass_of", "Trade"],
        ["Ban", "subclass_of", "Trade"],
        ["Ban", "subclass_of", "Sanction"],
        ["Embargo", "subclass_of", "Sanction"],
    ],
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    from hontology.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    (tmp_path / "a.txt").write_text("A tariff was imposed on steel.")
    with session_scope() as session:
        ontology = service.import_ontology(session, PAYLOAD)
        ids = {c.name: c.id for c in service.list_concepts(session, ontology.id)}
        documents = [
            Document(url=f"https://ex.test/{i}", url_hash=f"doclabel{i:04d}", title=f"Doc {i}")
            for i in range(2)
        ]
        documents[0].body_path = "a.txt"
        session.add_all(documents)
        session.flush()
        return {"ontology": ontology.id, "ids": ids, "documents": [d.id for d in documents]}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_leaves_come_with_their_top_level_families(client, world):
    leaves = client.get("/labels/leaves", params={"ontology_id": world["ontology"]}).json()
    families = {leaf["name"]: leaf["families"] for leaf in leaves}
    assert families == {
        "Tariff": ["Trade"],
        "Ban": ["Sanction", "Trade"],
        "Embargo": ["Sanction"],
        "Hack": ["Hack"],
    }
    assert all(leaf["inclusion_criteria"] for leaf in leaves)


def test_saving_a_document_answers_every_leaf(client, world):
    ids, document = world["ids"], world["documents"][0]
    response = client.put(
        f"/labels/documents/{document}",
        json={"ontology_id": world["ontology"], "concept_ids": [ids["Tariff"]], "note": "x"},
    )
    assert response.json() == {"positives": 1, "negatives": 3}
    shown = client.get(
        f"/labels/documents/{document}", params={"ontology_id": world["ontology"]}
    ).json()
    assert shown["labelled"] is True
    assert shown["positives"] == [ids["Tariff"]]
    assert shown["note"] == "x"
    assert shown["body"] == "A tariff was imposed on steel."


def test_saving_none_labels_every_leaf_negative(client, world):
    document = world["documents"][0]
    client.put(
        f"/labels/documents/{document}",
        json={"ontology_id": world["ontology"], "concept_ids": []},
    )
    status = client.post(
        "/labels/documents/status",
        json={"ontology_id": world["ontology"], "document_ids": world["documents"]},
    ).json()
    assert [(s["document_id"], s["labelled"], s["positives"]) for s in status] == [
        (world["documents"][0], True, []),
        (world["documents"][1], False, []),
    ]


def test_relabelling_replaces_the_answer(client, world):
    ids, document = world["ids"], world["documents"][0]
    for chosen in (["Tariff"], ["Embargo", "Hack"]):
        client.put(
            f"/labels/documents/{document}",
            json={"ontology_id": world["ontology"], "concept_ids": [ids[n] for n in chosen]},
        )
    shown = client.get(
        f"/labels/documents/{document}", params={"ontology_id": world["ontology"]}
    ).json()
    assert sorted(shown["positives"]) == sorted([ids["Embargo"], ids["Hack"]])


def test_an_internal_class_cannot_be_labelled(client, world):
    response = client.put(
        f"/labels/documents/{world['documents'][0]}",
        json={"ontology_id": world["ontology"], "concept_ids": [world["ids"]["Trade"]]},
    )
    assert response.status_code == 422


def test_a_machine_proposal_does_not_mark_a_document_labelled(client, world):
    with session_scope() as session:
        for concept_id in (world["ids"][n] for n in ("Tariff", "Ban", "Embargo", "Hack")):
            label_service.upsert_label(
                session,
                document_id=world["documents"][1],
                concept_id=concept_id,
                matched=False,
                source=label_service.MACHINE,
            )
    status = client.post(
        "/labels/documents/status",
        json={"ontology_id": world["ontology"], "document_ids": [world["documents"][1]]},
    ).json()
    assert status[0]["labelled"] is False
