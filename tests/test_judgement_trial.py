"""Judgement trials: asked as a run asks, and never recorded.

A trial must send what the run would send, with only the edits it was given,
and it must leave no trace: no verdict, and no change to the class it rewords.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from hontology.api.main import app
from hontology.db.models import Concept, Document, PairLabel, Run, Verdict
from hontology.db.session import session_scope
from hontology.judge import prompts, trial
from hontology.judge.providers.base import Completion
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


class FakeJudge:
    """Says match exactly when the class wording mentions vessels; records calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, system, prompt, config, want_json, want_reasoning, model):
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        matched = "vessel" in prompt.split("=== Concept ===")[1]
        return Completion(
            text=json.dumps({"matched": matched, "confidence": 0.8, "evidence": ""}),
            latency_s=0.1,
        )


@pytest.fixture
def judge(monkeypatch):
    fake = FakeJudge()
    monkeypatch.setattr(trial, "get_provider", lambda name, routing=None: fake)
    return fake


@pytest.fixture
def world():
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-trial", name="Trial")
        strike = service.create_concept(
            session,
            ontology.id,
            name="Port strike",
            definition="Dockworkers stop work.",
            exclusion_criteria="Threats before a walkout.",
        )
        run = Run(
            name="trial",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={
                "judge": {"provider": "llamacpp", "model": "gemma", "prompt_id": "strict_v1"}
            },
            candidates_key="c",
            judge_key="j",
            status="done",
        )
        session.add(run)
        docs = []
        for key in ("yes", "said", "no", "other"):
            document = Document(url=f"https://trial.test/{key}", url_hash=f"trialtest0{key}")
            session.add(document)
            session.flush()
            docs.append(document.id)
        yes, said, no, other = docs
        session.add(
            PairLabel(
                document_id=yes,
                concept_id=strike.id,
                matched=True,
                source="human",
                ontology_version="v1",
            )
        )
        session.add(
            PairLabel(
                document_id=no,
                concept_id=strike.id,
                matched=False,
                source="human",
                ontology_version="v1",
            )
        )
        for document_id, matched in ((said, True), (no, False), (other, False)):
            session.add(
                Verdict(
                    run_id=run.id,
                    document_id=document_id,
                    concept_id=strike.id,
                    matched=matched,
                    confidence=0.9,
                    prompt_id="strict_v1",
                )
            )
        session.flush()
        return {
            "run": run.id,
            "ontology": ontology.id,
            "concept": strike.id,
            "docs": dict(zip(("yes", "said", "no", "other"), docs, strict=True)),
        }


def _trial(world, **kw) -> trial.Trial:
    return trial.Trial(
        run_id=world["run"],
        document_id=world["docs"]["said"],
        concept_id=world["concept"],
        **kw,
    )


def test_the_original_is_the_templates_text_and_the_saved_wording(world):
    with session_scope() as session:
        rendered = trial.render(session, _trial(world))
    assert rendered["system"] == prompts.get("strict_v1").system
    assert "does NOT count when: Threats before a walkout." in rendered["prompt"]


def test_edits_reach_the_prompt_but_not_the_class(world):
    edit = {"exclusion_criteria": "Attacks on vessels."}
    with session_scope() as session:
        rendered = trial.render(session, _trial(world, system="Be terse.", wording=edit))
    assert rendered["system"] == "Be terse."
    assert "does NOT count when: Attacks on vessels." in rendered["prompt"]
    assert "definition: Dockworkers stop work." in rendered["prompt"]
    with session_scope() as session:
        saved = session.get(Concept, world["concept"])
        assert saved.exclusion_criteria == "Threats before a walkout."


def test_asking_uses_the_runs_model_and_records_nothing(world, judge):
    with session_scope() as session:
        before = session.scalar(select(func.count(Verdict.id)))
        original = trial.ask(session, _trial(world))
        edited = trial.ask(session, _trial(world, wording={"exclusion_criteria": "vessels"}))
        after = session.scalar(select(func.count(Verdict.id)))
    assert (original["matched"], edited["matched"]) == (False, True)
    assert [call["model"] for call in judge.calls] == ["gemma", "gemma"]
    assert before == after


def test_only_per_pair_prompts_can_be_tried(world):
    assert "strict_batch_v1" not in [t["prompt_id"] for t in trial.per_pair_templates()]
    with session_scope() as session, pytest.raises(ValueError):
        trial.render(session, _trial(world, prompt_id="strict_batch_v1"))


def test_articles_come_most_informative_first(world):
    client = TestClient(app)
    response = client.post(
        "/judgement/articles", json={"run_id": world["run"], "concept_id": world["concept"]}
    )
    assert response.status_code == 200, response.text
    docs = world["docs"]
    order = [a["document_id"] for a in response.json()]
    # Labelled match, then what the run said yes to, then labelled and judged.
    assert order == [docs["yes"], docs["said"], docs["no"], docs["other"]]
    by_id = {a["document_id"]: a for a in response.json()}
    assert by_id[docs["no"]]["label"] is False
    assert by_id[docs["no"]]["verdict"]["matched"] is False


def test_the_ask_endpoint(world, judge):
    client = TestClient(app)
    payload = {
        "run_id": world["run"],
        "document_id": world["docs"]["said"],
        "concept_id": world["concept"],
        "wording": {"definition": "Seizing a vessel."},
    }
    body = client.post("/judgement/ask", json=payload).json()
    assert body["matched"] is True
    assert body["model"] == "gemma"
    bad = client.post("/judgement/ask", json=payload | {"wording": {"name": "x"}})
    assert bad.status_code == 422


def test_pasted_text_is_asked_about_and_never_stored(world, judge):
    from hontology.db.models import Document

    with session_scope() as session:
        documents = session.scalar(select(func.count(Document.id)))
        verdicts = session.scalar(select(func.count(Verdict.id)))
        answer = trial.ask(
            session,
            trial.Trial(
                run_id=world["run"],
                document_id=None,
                concept_id=world["concept"],
                text="Dockworkers walked out at the port. " * 200,
                title="A strike",
                wording={"exclusion_criteria": "vessels"},
            ),
        )
        assert session.scalar(select(func.count(Document.id))) == documents
        assert session.scalar(select(func.count(Verdict.id))) == verdicts
    assert answer["matched"] is True
    assert "title: A strike" in answer["prompt"]
    # Cut to the run's article length, as a stored article's text would be.
    body = answer["prompt"].split("body: ", 1)[1].split("\n\n=== Concept", 1)[0]
    assert len(body) <= 3000


def test_the_ask_endpoint_takes_pasted_text(world, judge):
    client = TestClient(app)
    payload = {"run_id": world["run"], "concept_id": world["concept"], "text": "A port strike."}
    assert client.post("/judgement/ask", json=payload).status_code == 200
    empty = client.post("/judgement/ask", json=payload | {"text": "   "})
    assert empty.status_code == 422
    neither = client.post(
        "/judgement/ask", json={"run_id": world["run"], "concept_id": world["concept"]}
    )
    assert neither.status_code == 422
