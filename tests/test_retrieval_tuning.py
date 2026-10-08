"""Re-cutting a stored pool: the same rule as at build time, and honest recall.

A cutoff re-applied to a run's pool must keep exactly what the run kept when
given the run's own settings, or every comparison on the Retrieval page is
against a phantom. Recall must split true matches the pool never ranked from
those the cutoff dropped, since only the second kind is a cutoff's to fix.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hontology.api.main import app
from hontology.db.models import Candidate, Document, PairLabel, Run
from hontology.db.session import session_scope
from hontology.ontology import service
from hontology.retrieve import candidates, tuning

pytestmark = pytest.mark.requires_db

CONFIG = {
    "candidates": {"selection": "adaptive", "min_score": 0.5, "rel_margin": 0.05, "max_k": 2}
}
# Per article: (class, score), best first.
POOLS = {
    "a": [("Strike", 0.80), ("Flood", 0.77), ("Riot", 0.74), ("Fire", 0.40)],
    "b": [("Riot", 0.60), ("Fire", 0.58), ("Strike", 0.30)],
    "c": [("Flood", 0.45), ("Fire", 0.44)],
}


@pytest.fixture
def world():
    """A run whose stored pool and selection came from the real selection rule."""
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-tuning", name="Tuning")
        ids = {
            name: service.create_concept(session, ontology.id, name=name, definition=name).id
            for name in ("Strike", "Flood", "Riot", "Fire", "Never")
        }
        run = Run(
            name="tuning",
            ontology_id=ontology.id,
            ontology_version="v1",
            config=CONFIG,
            candidates_key="c",
            judge_key="j",
            status="done",
        )
        session.add(run)
        documents = {}
        for key in POOLS:
            document = Document(url=f"https://tune.test/{key}", url_hash=f"tunetest000{key}")
            session.add(document)
            session.flush()
            documents[key] = document.id
        own = tuning.run_cutoff(run)
        for key, pool in POOLS.items():
            ranked = [(ids[name], score) for name, score in pool]
            chosen = {
                cid
                for cid, _ in candidates.select_adaptive(
                    ranked, min_score=own.min_score, rel_margin=own.rel_margin, max_k=own.max_k
                )
            }
            for rank, (concept_id, score) in enumerate(ranked, start=1):
                session.add(
                    Candidate(
                        run_id=run.id,
                        document_id=documents[key],
                        concept_id=concept_id,
                        source="semantic",
                        score=score,
                        rank=rank,
                        selected=concept_id in chosen,
                    )
                )
        # True matches: Strike on a (kept), Riot on a (in the pool, cut),
        # Never on b (not in the pool at all). One labelled negative: Flood on a.
        for key, name, matched in (
            ("a", "Strike", True),
            ("a", "Riot", True),
            ("b", "Never", True),
            ("a", "Flood", False),
        ):
            session.add(
                PairLabel(
                    document_id=documents[key],
                    concept_id=ids[name],
                    matched=matched,
                    source="human",
                    ontology_version="v1",
                )
            )
        session.flush()
        return {"run": run.id, "ontology": ontology.id, "ids": ids, "documents": documents}


def _report(world, cutoff: tuning.Cutoff) -> dict:
    with session_scope() as session:
        labels = tuning.truth(session, world["ontology"])
        return tuning.report(session, world["run"], cutoff, labels)


def test_the_runs_own_cutoff_keeps_what_the_run_kept(world):
    with session_scope() as session:
        run = session.get(Run, world["run"])
        own = tuning.run_cutoff(run)
        stored = sum(
            1 for c in session.query(Candidate).filter_by(run_id=world["run"]) if c.selected
        )
    report = _report(world, own)
    # a: Strike, Flood (Riot is outside the 0.05 margin); b: Riot, Fire; c: below 0.5.
    assert report["pairs_kept"] == stored == 4
    assert report["documents_with_none"] == 1
    assert report["kept_per_document"] == {0: 1, 2: 2}


def test_recall_separates_the_cutoff_from_the_ranking(world):
    labels = _report(world, tuning.Cutoff(min_score=0.5, rel_margin=0.05, max_k=2))["labels"]
    assert labels["positives"] == 3
    assert labels["positives_in_pool"] == 2  # "Never" was not ranked
    assert labels["positives_kept"] == 1  # Riot on a was ranked but cut
    assert (labels["kept_labelled"], labels["kept_positive"]) == (2, 1)

    looser = _report(world, tuning.Cutoff(min_score=0.5, rel_margin=0.10, max_k=3))["labels"]
    assert looser["positives_kept"] == 2


def test_top_k_is_a_fixed_count(world):
    assert _report(world, tuning.Cutoff(selection="top-k", top_k=1))["pairs_kept"] == 3


def test_an_article_shows_both_cutoffs_and_what_the_pool_missed(world):
    with session_scope() as session:
        labels = tuning.truth(session, world["ontology"])
        pool = tuning.document_pool(
            session,
            world["run"],
            world["documents"]["b"],
            tuning.Cutoff(selection="top-k", top_k=1),
            labels,
        )
    assert [(r["name"], r["selected"], r["kept"]) for r in pool["pool"]] == [
        ("Riot", True, True),
        ("Fire", True, False),
        ("Strike", False, False),
    ]
    assert pool["missed"] == ["Never"]


def test_the_api_reports_a_setting_beside_the_runs_own(world):
    client = TestClient(app)
    response = client.post(
        f"/retrieval/runs/{world['run']}/report",
        json={"cutoff": {"selection": "top-k", "top_k": 1}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["setting"]["pairs_kept"], body["run"]["pairs_kept"]) == (3, 4)

    runs = client.get("/retrieval/runs", params={"ontology_id": world["ontology"]}).json()
    assert [(r["id"], r["documents"]) for r in runs] == [(world["run"], 3)]
    assert runs[0]["cutoff"]["max_k"] == 2


def test_a_labels_file_replaces_the_bank(world):
    csv_text = "document_url,concepts\nhttps://tune.test/a,Riot\nhttps://tune.test/b,none\n"
    client = TestClient(app)
    body = client.post(
        f"/retrieval/runs/{world['run']}/report", json={"labels_csv": csv_text}
    ).json()
    labels = body["run"]["labels"]
    assert (labels["documents"], labels["positives"], labels["positives_kept"]) == (2, 1, 0)

    bad = client.post(
        f"/retrieval/runs/{world['run']}/report",
        json={"labels_csv": "document_url,concepts\nhttps://tune.test/a,Volcano\n"},
    )
    assert bad.status_code == 422
