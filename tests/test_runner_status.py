"""A run's status row must end in a terminal state however execution ends.

A row left ``running`` by a crash is indistinguishable from a run in progress,
and nothing ever clears it — which is how several did accumulate.
"""

from __future__ import annotations

import pytest

from hontology.db.models import Run
from hontology.db.session import session_scope
from hontology.evalkit import runner
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


@pytest.fixture
def run_id():
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-status", name="Status")
        service.create_concept(session, ontology.id, name="Riot", definition="A riot.")
        run = runner.create_run(session, ontology_id=ontology.id, config={}, name="status")
        return run.id


def _execute(run_id: int, **kwargs) -> Run:
    with session_scope() as session:
        runner.execute(session, session.get(Run, run_id), **kwargs)


def _row(run_id: int) -> Run:
    with session_scope() as session:
        run = session.get(Run, run_id)
        session.expunge(run)
        return run


def _no_candidates(*args, **kwargs):
    return {"pool_rows": 0}


class TestTerminalStatus:
    def test_a_retrieval_failure_is_recorded(self, run_id, monkeypatch):
        """Retrieval used to run outside the failure handler entirely."""

        def boom(*args, **kwargs):
            raise RuntimeError("embedding provider unreachable")

        monkeypatch.setattr(runner.candidates_module, "build_candidates", boom)
        with pytest.raises(RuntimeError):
            _execute(run_id)

        row = _row(run_id)
        assert row.status == "failed"
        assert row.stage == "candidates"
        assert "unreachable" in row.error
        assert row.finished_at is not None

    def test_an_interrupt_is_recorded(self, run_id, monkeypatch):
        """Ctrl-C is not an Exception, and was not caught at all."""

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runner.candidates_module, "build_candidates", _no_candidates)
        monkeypatch.setattr(runner.judge_run_module, "judge_run", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _execute(run_id)

        row = _row(run_id)
        assert (row.status, row.stage, row.error) == ("failed", "judge", "interrupted")

    def test_retrieval_only_does_not_look_mid_judge(self, run_id, monkeypatch):
        monkeypatch.setattr(runner.candidates_module, "build_candidates", _no_candidates)
        _execute(run_id, skip_judge=True)

        row = _row(run_id)
        assert (row.status, row.stage) == ("candidates", None)

    def test_a_completed_run_clears_its_stage(self, run_id, monkeypatch):
        monkeypatch.setattr(runner.candidates_module, "build_candidates", _no_candidates)
        monkeypatch.setattr(runner.judge_run_module, "judge_run", lambda *a, **k: {"total": 0})
        monkeypatch.setattr(runner.judge_run_module, "liveness", lambda *a, **k: {"ok": True})
        _execute(run_id)

        row = _row(run_id)
        assert (row.status, row.stage) == ("done", None)
