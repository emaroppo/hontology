"""Similarity proposals over GKG themes, and the endpoints the Ontology page uses.

Most of the theme vocabulary names entities, not events, and a theme's stored
name is its code plus a usage count. Proposals must consider only event-like,
common themes, embedded as readable words, or a similarity run ranks
``TAX_WORLDFISH_ROACH`` against a port strike.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from hontology.apps.api.main import app
from hontology.db.models import Code, ConceptCode, SimilarityScore
from hontology.db.session import session_scope
from hontology.ontology import service
from hontology.pipeline.ingest.codes import themes
from hontology.pipeline.retrieve import embed, similarity

LOOKUP = [
    ("WB_167_PORTS", 24_531_937),
    ("STRIKE", 5_000_000),
    ("MANMADE_DISASTER_POWER_OUTAGE", 50_000),
    ("TAX_FNCACT_PRESIDENT", 9_999_999),  # an entity list: never proposed
    ("ROAD_INCIDENT_BRIDGE_COLLAPSE", 282),  # too rare to propose
]


class FakeEmbedder:
    """Deterministic three-dimensional vectors, so a run needs no model."""

    name = "fake"

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([digest[0] / 255 + 0.01, digest[1] / 255 + 0.01, digest[2] / 255 + 0.01])
        return out

    def dimension(self, model: str) -> int:
        return 3


class TestThemeText:
    def test_opaque_prefixes_and_underscores_go(self):
        assert themes.readable("WB_167_PORTS") == "ports"
        assert themes.readable("CRISISLEX_T06_SUPPLIES") == "supplies"
        assert (
            themes.readable("MANMADE_DISASTER_POWER_OUTAGE") == "manmade disaster power outage"
        )

    def test_the_usage_count_is_read_from_the_name(self):
        assert themes.uses("WB_167_PORTS (24,531,937 uses)") == 24_531_937
        assert themes.uses("no count here") is None
        assert themes.uses(None) is None

    def test_entity_lists_and_rare_themes_are_not_proposable(self):
        assert themes.proposal_text("STRIKE", "STRIKE (5,000,000 uses)") == "strike"
        assert themes.proposal_text("TAX_FNCACT_PRESIDENT", "x (9,999,999 uses)") is None
        assert themes.proposal_text("ROAD_INCIDENT_BRIDGE_COLLAPSE", "x (282 uses)") is None


@pytest.fixture
def loaded():
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-themes", name="Themes")
        strike = service.create_concept(
            session, ontology.id, name="Port strike", definition="Dockworkers stop work."
        )
        result = themes.load_themes(session, LOOKUP, source_url="test")
        codes = {
            c.code: c.id
            for c in session.scalars(select(Code).where(Code.system_id == result["system_id"]))
        }
        return {
            "ontology": ontology.id,
            "system": result["system_id"],
            "strike": strike.id,
            "codes": codes,
        }


@pytest.mark.requires_db
class TestThemeRuns:
    def test_only_proposable_themes_are_embedded_as_words(self, loaded):
        with session_scope() as session:
            texts = similarity.proposal_texts(
                session, system_id=loaded["system"], level="theme"
            )
        by_code = {code: texts.get(code_id) for code, code_id in loaded["codes"].items()}
        assert by_code == {
            "WB_167_PORTS": "ports",
            "STRIKE": "strike",
            "MANMADE_DISASTER_POWER_OUTAGE": "manmade disaster power outage",
            "TAX_FNCACT_PRESIDENT": None,
            "ROAD_INCIDENT_BRIDGE_COLLAPSE": None,
        }

    def test_a_run_without_auto_link_only_scores(self, loaded):
        with session_scope() as session:
            result = similarity.run_similarity(
                session,
                FakeEmbedder(),
                "fake",
                ontology_id=loaded["ontology"],
                system_id=loaded["system"],
                level="theme",
                adaptive=True,
                auto_link=False,
            )
            assert result.n_linked == 0
            assert session.scalars(select(ConceptCode)).all() == []
            scored = set(
                session.scalars(
                    select(SimilarityScore.target_id).where(
                        SimilarityScore.run_id == result.run_id
                    )
                )
            )
        proposable = {"WB_167_PORTS", "STRIKE", "MANMADE_DISASTER_POWER_OUTAGE"}
        assert scored == {loaded["codes"][code] for code in proposable}

    def test_a_hand_made_link_to_a_rare_theme_survives_a_run(self, loaded):
        rare = loaded["codes"]["ROAD_INCIDENT_BRIDGE_COLLAPSE"]
        with session_scope() as session:
            similarity.set_link(session, loaded["strike"], rare, linked=True)
            similarity.run_similarity(
                session,
                FakeEmbedder(),
                "fake",
                ontology_id=loaded["ontology"],
                system_id=loaded["system"],
                level="theme",
                adaptive=True,
            )
            links = {
                row.code_id: row.similarity_run_id
                for row in session.scalars(
                    select(ConceptCode).where(ConceptCode.concept_id == loaded["strike"])
                )
            }
        assert links[rare] is None
        assert loaded["codes"]["TAX_FNCACT_PRESIDENT"] not in links


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.requires_db
class TestApi:
    def test_systems_list_their_levels(self, client, loaded):
        systems = {s["slug"]: s for s in client.get("/taxonomy/systems").json()}
        assert systems["gkg-themes"]["levels"] == [{"level": "theme", "n_codes": len(LOOKUP)}]

    def test_similarity_picks_the_level_and_links_nothing(self, client, loaded, monkeypatch):
        monkeypatch.setattr(embed, "get_provider", lambda name, **kw: FakeEmbedder())
        response = client.post(
            "/taxonomy/similarity",
            json={"ontology_id": loaded["ontology"], "system": "gkg-themes"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["n_scores"] == 3
        links = client.get(f"/taxonomy/ontologies/{loaded['ontology']}/links").json()
        assert links == {}

    def test_an_unloaded_system_is_a_conflict(self, client, loaded):
        response = client.post(
            "/taxonomy/similarity", json={"ontology_id": loaded["ontology"], "system": "nope"}
        )
        assert response.status_code == 409

    def test_links_come_back_per_class_with_their_system(self, client, loaded):
        client.post(
            "/taxonomy/links",
            json={
                "concept_id": loaded["strike"],
                "code_id": loaded["codes"]["WB_167_PORTS"],
                "linked": True,
            },
        )
        links = client.get(f"/taxonomy/ontologies/{loaded['ontology']}/links").json()
        [link] = links[str(loaded["strike"])]
        assert link["code"]["system"] == "gkg-themes"
        assert link["code"]["code"] == "WB_167_PORTS"
        assert link["manual"] is True


@pytest.mark.requires_db
def test_cameo_defaults_to_its_event_level(client):
    """CAMEO's 3-digit level holds more codes than its 4-digit one, so "most
    populous" would quietly coarsen every run; the default is named instead."""
    from hontology.pipeline.ingest.codes import cameo

    with session_scope() as session:
        cameo.load_codes(
            session,
            cameo.parse_lookup("14\tPROTEST\n145\tRiot\n146\tOther\n1451\tDissent, riot\n"),
            source_url="test",
        )
    systems = {s["slug"]: s for s in client.get("/taxonomy/systems").json()}
    assert [lv["level"] for lv in systems["cameo"]["levels"]] == ["event", "base", "root"]
