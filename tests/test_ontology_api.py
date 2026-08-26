"""Ontology API behaviour, against a live database.

Marked ``requires_db``: these exercise real Postgres constraints, which is the
point — the uniqueness and cascade rules are schema-level, and a mocked session
would assert nothing about them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hontology.api.main import app

pytestmark = pytest.mark.requires_db


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def make_ontology(client: TestClient, slug: str = "test-supply") -> int:
    response = client.post(
        "/ontologies", json={"slug": slug, "name": "Supply chain disruption"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_fetch_ontology(client: TestClient):
    ontology_id = make_ontology(client)
    fetched = client.get(f"/ontologies/{ontology_id}").json()
    assert fetched["slug"] == "test-supply"
    assert fetched["name"] == "Supply chain disruption"


def test_duplicate_slug_conflicts(client: TestClient):
    make_ontology(client)
    again = client.post("/ontologies", json={"slug": "test-supply", "name": "Other"})
    assert again.status_code == 409


def test_slug_must_be_url_safe(client: TestClient):
    bad = client.post("/ontologies", json={"slug": "Not A Slug", "name": "x"})
    assert bad.status_code == 422


def test_missing_ontology_is_404(client: TestClient):
    assert client.get("/ontologies/98765").status_code == 404


def test_concept_crud_round_trip(client: TestClient):
    ontology_id = make_ontology(client)

    created = client.post(
        f"/ontologies/{ontology_id}/concepts",
        json={
            "name": "Port closure",
            "definition": "A commercial seaport halts vessel operations.",
            "exclusion_criteria": "Planned or threatened closures.",
            "category": "logistics",
            "weight": 2.0,
        },
    )
    assert created.status_code == 201, created.text
    concept_id = created.json()["id"]
    assert created.json()["category_id"] is not None

    patched = client.patch(
        f"/ontologies/{ontology_id}/concepts/{concept_id}",
        json={"definition": "A seaport suspends vessel operations."},
    )
    assert patched.status_code == 200
    assert patched.json()["definition"] == "A seaport suspends vessel operations."
    # A patch of one field must not clear the others.
    assert patched.json()["exclusion_criteria"] == "Planned or threatened closures."

    listed = client.get(f"/ontologies/{ontology_id}/concepts").json()
    assert [c["name"] for c in listed] == ["Port closure"]

    assert client.delete(f"/ontologies/{ontology_id}/concepts/{concept_id}").status_code == 204
    assert client.get(f"/ontologies/{ontology_id}/concepts").json() == []


def test_duplicate_concept_name_conflicts(client: TestClient):
    ontology_id = make_ontology(client)
    body = {"name": "Port closure"}
    assert client.post(f"/ontologies/{ontology_id}/concepts", json=body).status_code == 201
    assert client.post(f"/ontologies/{ontology_id}/concepts", json=body).status_code == 409


def test_export_import_round_trip(client: TestClient):
    ontology_id = make_ontology(client)
    client.post(
        f"/ontologies/{ontology_id}/concepts",
        json={
            "name": "Port closure",
            "definition": "A seaport halts operations.",
            "category": "logistics",
        },
    )

    exported = client.get(f"/ontologies/{ontology_id}/export").json()
    assert exported["concepts"][0]["category"] == "logistics"

    exported["slug"] = "test-copy"
    imported = client.post("/ontologies/import", json=exported)
    assert imported.status_code == 200

    copied = client.get(f"/ontologies/{imported.json()['id']}/concepts").json()
    assert [c["name"] for c in copied] == ["Port closure"]


def test_reimport_merges_rather_than_duplicating(client: TestClient):
    """The property that makes the export safe to keep in version control."""
    ontology_id = make_ontology(client)
    client.post(
        f"/ontologies/{ontology_id}/concepts",
        json={"name": "Port closure", "definition": "Original wording."},
    )

    exported = client.get(f"/ontologies/{ontology_id}/export").json()
    exported["concepts"][0]["definition"] = "Edited wording."
    client.post("/ontologies/import", json=exported)

    concepts = client.get(f"/ontologies/{ontology_id}/concepts").json()
    assert len(concepts) == 1
    assert concepts[0]["definition"] == "Edited wording."


def test_snapshot_mints_only_on_change(client: TestClient):
    ontology_id = make_ontology(client)
    client.post(
        f"/ontologies/{ontology_id}/concepts",
        json={"name": "Port closure", "definition": "A seaport halts operations."},
    )

    first = client.post(f"/ontologies/{ontology_id}/snapshot").json()
    assert first["version"] == "v1"
    assert first["created"] is True

    # Nothing changed: same version, and nothing new minted.
    repeat = client.post(f"/ontologies/{ontology_id}/snapshot").json()
    assert repeat["version"] == "v1"
    assert repeat["created"] is False

    concepts = client.get(f"/ontologies/{ontology_id}/concepts").json()
    client.patch(
        f"/ontologies/{ontology_id}/concepts/{concepts[0]['id']}",
        json={"definition": "A seaport suspends all vessel movements."},
    )
    after_edit = client.post(f"/ontologies/{ontology_id}/snapshot").json()
    assert after_edit["version"] == "v2"
    assert after_edit["created"] is True


def test_weight_change_does_not_mint_a_version(client: TestClient):
    """Retuning scoring must not invalidate labels, so it must not fork a version."""
    ontology_id = make_ontology(client)
    created = client.post(
        f"/ontologies/{ontology_id}/concepts",
        json={
            "name": "Port closure",
            "definition": "A seaport halts operations.",
            "weight": 1.0,
        },
    ).json()

    baseline = client.post(f"/ontologies/{ontology_id}/snapshot").json()
    client.patch(f"/ontologies/{ontology_id}/concepts/{created['id']}", json={"weight": 42.0})
    after = client.post(f"/ontologies/{ontology_id}/snapshot").json()

    assert after["version"] == baseline["version"]
    assert after["created"] is False


def test_deleting_an_ontology_cascades_to_concepts(client: TestClient):
    ontology_id = make_ontology(client)
    client.post(f"/ontologies/{ontology_id}/concepts", json={"name": "Port closure"})

    assert client.delete(f"/ontologies/{ontology_id}").status_code == 204
    assert client.get(f"/ontologies/{ontology_id}").status_code == 404
