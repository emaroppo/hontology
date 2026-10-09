"""Run and adjudication endpoints.

Background execution is patched out here: the point is that the request returns
without waiting and the job record is correct, not that a model runs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hontology.apps.api.main import app
from hontology.db.models import Document, PairLabel
from hontology.db.session import session_scope
from hontology.evaluation.labels import bank
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def fixture(monkeypatch):
    """An ontology with one concept and one document; background work stubbed."""
    scheduled: list[tuple] = []

    def fake_background(run_id: int, **kwargs):
        scheduled.append((run_id, kwargs))

    monkeypatch.setattr(
        "hontology.apps.api.routers.runs._execute_in_background", fake_background
    )

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-runsapi", name="Runs API")
        concept = service.create_concept(
            session, ontology.id, name="Riot", definition="A violent disturbance."
        )
        document = Document(url="https://ra.test/1", url_hash="rahash000000001")
        session.add(document)
        session.flush()
        ids = {
            "ontology": ontology.id,
            "concept": concept.id,
            "document": document.id,
            "scheduled": scheduled,
        }
    yield ids


class TestStartRun:
    def test_returns_immediately_with_a_job_record(self, client, fixture):
        response = client.post(
            "/runs",
            json={"ontology_id": fixture["ontology"], "name": "async", "document_limit": 5},
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "pending"
        assert body["candidates_key"] and body["judge_key"]
        # The work was handed to the background, not done inline.
        assert fixture["scheduled"] and fixture["scheduled"][0][0] == body["id"]

    def test_the_run_is_pollable(self, client, fixture):
        run_id = client.post("/runs", json={"ontology_id": fixture["ontology"]}).json()["id"]

        polled = client.get(f"/runs/{run_id}")
        assert polled.status_code == 200
        assert polled.json()["id"] == run_id
        assert "progress_done" in polled.json()

    def test_an_invalid_config_is_rejected_before_scheduling(self, client, fixture):
        response = client.post(
            "/runs",
            json={
                "ontology_id": fixture["ontology"],
                "config": {"candidates": {"source": "telepathy"}},
            },
        )
        assert response.status_code == 422
        assert fixture["scheduled"] == []

    def test_a_missing_ontology_is_404(self, client, fixture):
        assert client.post("/runs", json={"ontology_id": 987654}).status_code == 404

    def test_polling_a_missing_run_is_404(self, client, fixture):
        assert client.get("/runs/987654").status_code == 404

    def test_resume_schedules_without_recreating(self, client, fixture):
        run_id = client.post("/runs", json={"ontology_id": fixture["ontology"]}).json()["id"]
        fixture["scheduled"].clear()

        response = client.post(f"/runs/{run_id}/resume")
        assert response.status_code == 202
        assert response.json()["id"] == run_id
        assert fixture["scheduled"][0][0] == run_id

    def test_manifest_records_infrastructure(self, client, fixture):
        run_id = client.post("/runs", json={"ontology_id": fixture["ontology"]}).json()["id"]
        manifest = client.get(f"/runs/{run_id}/manifest").json()
        assert "infra" in manifest
        assert "keys" in manifest


class TestKeyPreview:
    def test_prompt_change_preserves_the_candidates_key(self, client, fixture):
        strict = client.post(
            "/runs/keys", json={"config": {"judge": {"prompt_id": "strict_v1"}}}
        ).json()
        lenient = client.post(
            "/runs/keys", json={"config": {"judge": {"prompt_id": "lenient_v1"}}}
        ).json()

        assert strict["candidates"] == lenient["candidates"]
        assert strict["judge"] != lenient["judge"]

    def test_invalid_config_is_422(self, client, fixture):
        response = client.post("/runs/keys", json={"config": {"judge": {"samples": 0}}})
        assert response.status_code == 422


class TestAdjudicationEndpoints:
    def test_pending_lists_machine_proposals(self, client, fixture):
        with session_scope() as session:
            bank.upsert_label(
                session,
                document_id=fixture["document"],
                concept_id=fixture["concept"],
                matched=True,
                source=bank.MACHINE,
                proposed_by="some-model",
            )

        rows = client.get("/labels/pending", params={"ontology_id": fixture["ontology"]}).json()
        assert len(rows) == 1
        assert rows[0]["proposed_matched"] is True
        assert rows[0]["proposed_by"] == "some-model"
        assert rows[0]["concept_name"] == "Riot"

    def test_confirming_removes_it_from_pending_and_makes_it_count(self, client, fixture):
        with session_scope() as session:
            label_id = bank.upsert_label(
                session,
                document_id=fixture["document"],
                concept_id=fixture["concept"],
                matched=True,
                source=bank.MACHINE,
            ).id

        response = client.post(f"/labels/{label_id}/adjudicate", json={"matched": True})
        assert response.status_code == 200
        assert response.json()["source"] == "adjudicated"

        assert (
            client.get("/labels/pending", params={"ontology_id": fixture["ontology"]}).json()
            == []
        )
        stats = client.get("/labels/stats", params={"ontology_id": fixture["ontology"]}).json()
        assert stats["trusted"] == 1
        assert stats["pending_adjudication"] == 0

    def test_flipping_records_the_human_verdict(self, client, fixture):
        with session_scope() as session:
            label_id = bank.upsert_label(
                session,
                document_id=fixture["document"],
                concept_id=fixture["concept"],
                matched=True,
                source=bank.MACHINE,
            ).id

        client.post(f"/labels/{label_id}/adjudicate", json={"matched": False})
        with session_scope() as session:
            label = session.get(PairLabel, label_id)
            assert label.matched is False
            assert label.source == "adjudicated"

    def test_stale_detail_lists_rewordings(self, client, fixture):
        with session_scope() as session:
            bank.upsert_label(
                session,
                document_id=fixture["document"],
                concept_id=fixture["concept"],
                matched=True,
                source=bank.HUMAN,
            )
        with session_scope() as session:
            service.update_concept(
                session, fixture["concept"], definition="Reworded definition."
            )

        rows = client.get(
            "/labels/stale/detail", params={"ontology_id": fixture["ontology"]}
        ).json()
        assert len(rows) == 1
        assert rows[0]["stale"] is True


def test_options_offer_every_registered_prompt_and_the_defaults(client):
    from hontology.pipeline.judge import prompts
    from hontology.pipeline.runs import config as run_config

    body = client.get("/runs/options").json()
    assert [p["prompt_id"] for p in body["choices"]["prompts"]] == prompts.available()
    assert body["choices"]["aggregation"] == list(run_config.AGGREGATIONS)
    # The form starts from exactly what an empty config normalizes to.
    defaults = run_config.normalize({})
    assert body["defaults"] == {k: defaults[k] for k in ("common", "candidates", "judge")}


@pytest.mark.requires_db
def test_a_runs_config_comes_back_to_start_another_from(client, fixture):
    config = {"candidates": {"selection": "top-k", "top_k": 4}}
    started = client.post(
        "/runs", json={"ontology_id": fixture["ontology"], "config": config}
    ).json()
    stored = client.get(f"/runs/{started['id']}/config").json()
    assert stored["candidates"]["top_k"] == 4
    # Started again from it, a run resolves to the same stage keys.
    keys = client.post(
        "/runs/keys",
        json={"config": stored, "ontology_version": stored["common"]["ontology_version"]},
    ).json()
    assert keys == {"candidates": started["candidates_key"], "judge": started["judge_key"]}
    assert client.get("/runs/987654/config").status_code == 404
