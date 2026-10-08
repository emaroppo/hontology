"""The live leaderboard: who covered the sample, and the endpoint that scores them.

A run is comparable on a sample only over the articles it worked on itself, so
coverage must follow the run, not the retrieval it may have borrowed: a run that
reused another's candidates for a different sample covers none of this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hontology.api.main import app
from hontology.db.models import Candidate, Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit import leaderboard
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def world(tmp_path):
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-board", name="Board")
        strike = service.create_concept(session, ontology.id, name="Strike")
        docs = []
        for i in range(4):
            document = Document(url=f"https://lb.test/{i}", url_hash=f"lbtest00000{i}")
            session.add(document)
            session.flush()
            docs.append(document.id)

        def run(name: str, manifest: dict | None = None) -> Run:
            row = Run(
                name=name,
                ontology_id=ontology.id,
                ontology_version="v1",
                config={"judge": {"prompt_id": "strict_v1"}},
                manifest=manifest,
                candidates_key="c",
                judge_key=f"j{name}",
                status="done",
            )
            session.add(row)
            session.flush()
            return row

        own = run("own")
        for doc in docs:
            session.add(
                Candidate(
                    run_id=own.id,
                    document_id=doc,
                    concept_id=strike.id,
                    source="semantic",
                    score=0.9,
                    rank=1,
                    selected=True,
                )
            )
            session.add(
                Verdict(
                    run_id=own.id, document_id=doc, concept_id=strike.id, matched=doc != docs[3]
                )
            )
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(
            json.dumps({"order": [{"document_id": d} for d in docs], "order_sha256": "s"})
        )
        other_path = tmp_path / "other.json"
        other_path.write_text(
            json.dumps({"order": [{"document_id": 999}], "order_sha256": "o"})
        )
        arm = run(
            "arm",
            {
                "sample": {
                    "manifest": str(sample_path),
                    "judged_first": 2,
                    "candidates_from": own.id,
                }
            },
        )
        dev = run(
            "dev",
            {
                "sample": {
                    "manifest": str(other_path),
                    "judged_first": 1,
                    "candidates_from": own.id,
                }
            },
        )
        session.add(
            Verdict(run_id=dev.id, document_id=docs[0], concept_id=strike.id, matched=True)
        )
        session.flush()
        return {
            "ontology": ontology.id,
            "docs": docs,
            "runs": {"own": own.id, "arm": arm.id, "dev": dev.id},
            "manifest": str(sample_path),
        }


def test_coverage_follows_what_the_run_itself_worked_on(world):
    docs = set(world["docs"])
    with session_scope() as session:
        covered = {
            name: leaderboard.coverage(session, session.get(Run, run_id), docs)
            for name, run_id in world["runs"].items()
        }
    # The dev run judged an article of this sample, but under its own sample.
    assert covered == {"own": 4, "arm": 2, "dev": 0}


def test_the_live_leaderboard_scores_every_judged_run(world):
    from hontology.evalkit import annotations

    with session_scope() as session:
        annotations.import_set(
            session,
            world["ontology"],
            "test-board-annotator",
            "document_url,concepts\n"
            + "".join(f"https://lb.test/{i},Strike\n" for i in range(4)),
        )
    response = TestClient(app).post(
        "/eval/leaderboard/live",
        json={
            "ontology_id": world["ontology"],
            "manifest_path": world["manifest"],
            "annotator": "test-board-annotator",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["labelled_articles"] == 4
    rows = {r["run_id"]: r for r in body["rows"]}
    own = rows[world["runs"]["own"]]
    # Three of four true strikes found, nothing wrongly.
    assert (own["end_to_end"]["tp"], own["end_to_end"]["fn"], own["end_to_end"]["fp"]) == (
        3,
        1,
        0,
    )
    assert own["versions"]["retrieval"].startswith("r-")
    assert world["runs"]["arm"] not in rows  # judged nothing


def test_a_run_that_missed_the_sample_is_scored_only_when_asked(world):
    from hontology.evalkit import annotations, leaderboard

    with session_scope() as session:
        annotations.import_set(
            session,
            world["ontology"],
            "test-board-partial",
            "document_url,concepts\n"
            + "".join(f"https://lb.test/{i},Strike\n" for i in range(4)),
        )
        labels = annotations.truth(session, world["ontology"], "test-board-partial")
        manifest = json.loads(Path(world["manifest"]).read_text())
        quick = leaderboard.rows(session, world["ontology"], manifest, labels)
        full = leaderboard.rows(
            session, world["ontology"], manifest, labels, include_partial=True
        )
    dev = world["runs"]["dev"]
    by_run = {r["run_id"]: r for r in quick["rows"]}
    assert by_run[dev]["covered"] == 0 and by_run[dev]["end_to_end"] is None
    assert {r["run_id"]: r for r in full["rows"]}[dev]["end_to_end"] is not None
    # The table never shows judge-only scores, so the leaderboard skips them.
    assert by_run[world["runs"]["own"]]["judge_only"] is None
