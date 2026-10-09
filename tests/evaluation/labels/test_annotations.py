"""Machine annotation sets live beside the human label bank, never in it.

The two truths must stay apart: a machine set can cover the very articles a
person labelled, and scoring against one must never read the other.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hontology.apps.api.main import app
from hontology.db.models import Document
from hontology.db.session import session_scope
from hontology.evaluation.labels import annotations
from hontology.evaluation.labels.bank import upsert_label
from hontology.ontology import service

pytestmark = pytest.mark.requires_db

FILE = "document_url,concepts\nhttps://ann.test/a,Strike\nhttps://ann.test/b,none\nhttps://ann.test/c,\n"


@pytest.fixture
def world():
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-annotations", name="Ann")
        strike = service.create_concept(session, ontology.id, name="Strike")
        flood = service.create_concept(session, ontology.id, name="Flood")
        docs = {}
        for key in "abc":
            document = Document(url=f"https://ann.test/{key}", url_hash=f"anntest0000{key}")
            session.add(document)
            session.flush()
            docs[key] = document.id
        # A person disagrees with the annotator on a's Strike.
        upsert_label(
            session,
            document_id=docs["a"],
            concept_id=strike.id,
            matched=False,
            ontology_id=ontology.id,
        )
        return {"ontology": ontology.id, "strike": strike.id, "flood": flood.id, "docs": docs}


def test_a_set_answers_every_leaf_of_every_labelled_article(world):
    with session_scope() as session:
        report = annotations.import_set(
            session, world["ontology"], "bot", FILE, description="a bot"
        )
        truth = annotations.truth(session, world["ontology"], "bot")
    assert (
        report["articles"],
        report["pairs"],
        report["positives"],
        report["skipped_blank"],
    ) == (
        2,
        4,
        1,
        1,
    )
    docs = world["docs"]
    assert truth == {
        (docs["a"], world["strike"]): True,
        (docs["a"], world["flood"]): False,
        (docs["b"], world["strike"]): False,
        (docs["b"], world["flood"]): False,
    }


def test_the_two_truths_never_mix(world):
    with session_scope() as session:
        annotations.import_set(session, world["ontology"], "bot", FILE)
        human = annotations.truth(session, world["ontology"])
        machine = annotations.truth(session, world["ontology"], "bot")
    key = (world["docs"]["a"], world["strike"])
    assert (human[key], machine[key]) == (False, True)
    assert len(human) == 1


def test_reimporting_extends_and_replaces(world):
    with session_scope() as session:
        annotations.import_set(session, world["ontology"], "bot", FILE)
        annotations.import_set(
            session,
            world["ontology"],
            "bot",
            "document_url,concepts\nhttps://ann.test/c,Flood\n",
        )
        truth = annotations.truth(session, world["ontology"], "bot")
    assert len(truth) == 6
    assert truth[(world["docs"]["c"], world["flood"])] is True


def test_an_unreadable_file_writes_nothing(world):
    with session_scope() as session:
        with pytest.raises(ValueError):
            annotations.import_set(
                session,
                world["ontology"],
                "bot",
                FILE + "https://ann.test/a,Volcano\n",
            )
        assert annotations.list_sets(session, world["ontology"]) == []


def test_the_truths_endpoint_lists_both_kinds(world):
    with session_scope() as session:
        annotations.import_set(session, world["ontology"], "bot", FILE, description="a bot")
    body = (
        TestClient(app).get("/labels/truths", params={"ontology_id": world["ontology"]}).json()
    )
    assert body["human"] == {"articles": 1, "pairs": 1, "positives": 0}
    assert body["machine"] == [
        {"name": "bot", "description": "a bot", "articles": 2, "pairs": 4, "positives": 1}
    ]
