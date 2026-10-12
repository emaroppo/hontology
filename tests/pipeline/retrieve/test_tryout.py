"""Trying a strategy through the API: what it would show the judge, nothing stored.

The embedding ranking is replaced by a fixed one, and the judge model by the
hierarchical tests' fake, so each test says exactly what is selected and routed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from hontology.apps.api.main import app
from hontology.db.models import Verdict
from hontology.db.session import session_scope
from hontology.pipeline.judge import run as judge_module
from hontology.pipeline.retrieve import pasted
from tests.pipeline.judge.test_hierarchical_judging import FakeJudge, world  # noqa: F401

pytestmark = pytest.mark.requires_db


@pytest.fixture
def tried(world, monkeypatch):  # noqa: F811
    provider = FakeJudge(world["names"], yes={"Trade"})
    monkeypatch.setattr(judge_module, "get_provider", lambda name, **kw: provider)
    kept = {"value": [world["ids"]["Ban"]]}

    def rank_text(session, run, body, cutoff, *, pool_size=20):
        return {"ranking": [{"concept_id": c, "kept": True} for c in kept["value"]]}

    monkeypatch.setattr(pasted, "rank_text", rank_text)

    def post(**payload):
        body = {"run_id": world["run"], "prompt_id": "hier_batch_v2"} | payload
        return TestClient(app).post("/retrieval/strategy", json=body)

    return post, provider, kept


def test_routing_on_pasted_text_shows_what_the_judge_would_see(world, tried):  # noqa: F811
    post, provider, _ = tried
    response = post(text="A government raised tariffs.")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["steps"] == ["embedding", "routing"]
    assert result["shown"] == ["Ban", "Hack", "Tariff"]
    assert result["routing"] == [
        {"asked": ["Trade", "Sanction"], "yes": ["Trade"], "error": None}
    ]
    assert result["spend"]["calls"] == 1 and provider.calls == [{"Trade", "Sanction"}]
    with session_scope() as session:
        assert session.scalar(select(func.count(Verdict.id))) == 0


def test_a_corpus_article_is_retrieved_from_its_stored_text(world, tried):  # noqa: F811
    post, _, _ = tried
    result = post(document_id=world["document"]).json()
    assert result["shown"] == ["Ban", "Hack", "Tariff"]


def test_an_article_the_cutoff_keeps_nothing_for_is_stopped_at_the_gate(world, tried):  # noqa: F811
    post, provider, kept = tried
    kept["value"] = []
    result = post(text="Nothing relevant.").json()
    assert result["stopped"] == "no leaf selected" and result["shown"] == []
    assert provider.calls == []


def test_only_prompts_whose_routing_runs_apart_are_offered(world):  # noqa: F811
    listed = {s["prompt_id"] for s in TestClient(app).get("/retrieval/strategies").json()}
    assert {"hier_batch_v2", "extract_v1", "extract_embed_v1"} <= listed
    assert "hier_batch_v1" not in listed
