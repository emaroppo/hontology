"""A run that stops early says so on its row.

A run left "running" by a process that died is indistinguishable from one still
in progress, and nothing ever clears it. SIGTERM is the case that slipped
through: Python ends on it without raising, so no handler recorded anything.
"""

from __future__ import annotations

import json
import os
import signal
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from hontology.cli import app, run_signals
from hontology.db.models import Run
from hontology.db.session import session_scope
from hontology.ontology import service


def test_sigterm_raises_like_ctrl_c():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        run_signals()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.fixture
def resumable(tmp_path, monkeypatch):
    """A finished run about to be extended, and the files `run sample` reads."""
    from hontology.evalkit import calendar_run

    monkeypatch.setattr(calendar_run, "check_same_leaves", lambda *a, **kw: None)
    stale = datetime(2026, 10, 7, 13, 11, tzinfo=UTC)
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-interrupt", name="Interrupt")
        runs = []
        for name in ("source", "arm"):
            run = Run(
                name=name,
                ontology_id=ontology.id,
                ontology_version="v1",
                config={},
                candidates_key="c",
                judge_key=f"j_{name}",
                status="done",
                finished_at=stale,
                error="left from the last pass",
            )
            session.add(run)
            session.flush()
            runs.append(run.id)
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({"order": [{"document_id": 1}], "order_sha256": "x"}))
    config = tmp_path / "config.json"
    config.write_text("{}")
    source, arm = runs
    args = [
        "run",
        "sample",
        str(ontology.id),
        str(config),
        str(manifest),
        "--candidates-from",
        str(source),
        "--run-id",
        str(arm),
    ]
    previous = signal.getsignal(signal.SIGTERM)
    yield {"run": arm, "args": args, "stale": stale}
    # Invoking a run command installs its SIGTERM handler for the whole process.
    signal.signal(signal.SIGTERM, previous)


def _row(run_id: int) -> Run:
    with session_scope() as session:
        return session.get(Run, run_id)


@pytest.mark.requires_db
def test_a_failed_sample_pass_is_recorded(resumable, monkeypatch):
    from hontology.evalkit import calendar_run

    def boom(*args, **kwargs):
        raise RuntimeError("judge provider went away")

    monkeypatch.setattr(calendar_run, "judge_documents", boom)
    result = CliRunner().invoke(app, resumable["args"])
    assert result.exit_code != 0

    row = _row(resumable["run"])
    assert row.status == "failed"
    assert row.error == "judge provider went away"
    assert row.finished_at > resumable["stale"]


@pytest.mark.requires_db
def test_a_resumed_pass_forgets_the_last_ones_finish(resumable, monkeypatch):
    """While it runs, the row must not claim it finished at the previous pass."""
    from hontology.evalkit import calendar_run

    seen = {}

    def judge(session, run, **kwargs):
        seen["finished_at"], seen["error"] = run.finished_at, run.error
        return {"retrieved": 1, "documents": 1, "judge": {"judged": 2, "matched": 1}}

    monkeypatch.setattr(calendar_run, "judge_documents", judge)
    result = CliRunner().invoke(app, resumable["args"])
    assert result.exit_code == 0, result.output

    assert seen == {"finished_at": None, "error": None}
    row = _row(resumable["run"])
    assert row.status == "done"
    assert row.finished_at > resumable["stale"]
