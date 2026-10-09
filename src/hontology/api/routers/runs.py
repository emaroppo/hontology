"""Run endpoints: start a run, watch it, resume it.

A run takes minutes, so `POST /runs` returns immediately with an id and the work
continues in the background. The `Run` row is the job record — status, stage and
progress live on it — so a process restart loses the *reporting*, not the run:
verdicts commit per pair, and `POST /runs/{id}/resume` picks up exactly where it
stopped.

Background tasks open their own session. The request's session is closed the
moment the response is sent, so reusing it here would fail on the first query
after the client disconnects.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.api.routers._common import run_or_404
from hontology.db.models import Run
from hontology.db.session import get_db, session_scope
from hontology.evalkit import config as run_config
from hontology.evalkit import runner

log = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


class RunIn(BaseModel):
    ontology_id: int
    config: dict = Field(default_factory=dict)
    name: str | None = None
    document_limit: int = Field(default=100, ge=1, le=5000)
    judge_limit: int | None = None
    skip_judge: bool = False


class RunOut(BaseModel):
    id: int
    name: str
    status: str
    stage: str | None = None
    ontology_id: int
    ontology_version: str
    candidates_key: str
    judge_key: str
    progress_done: int
    progress_total: int
    error: str | None = None

    @staticmethod
    def of(run: Run) -> RunOut:
        return RunOut(
            id=run.id,
            name=run.name,
            status=run.status,
            stage=run.stage,
            ontology_id=run.ontology_id,
            ontology_version=run.ontology_version,
            candidates_key=run.candidates_key,
            judge_key=run.judge_key,
            progress_done=run.progress_done,
            progress_total=run.progress_total,
            error=run.error,
        )


class KeysIn(BaseModel):
    config: dict = Field(default_factory=dict)
    ontology_version: str = "v1"


def _execute_in_background(
    run_id: int, *, document_limit: int, judge_limit: int | None, skip_judge: bool
) -> None:
    """Worker body. Opens its own session; never raises into the event loop."""
    try:
        with session_scope() as session:
            run = session.get(Run, run_id)
            if run is None:
                return
            runner.execute(
                session,
                run,
                document_limit=document_limit,
                judge_limit=judge_limit,
                skip_judge=skip_judge,
            )
    except Exception as exc:  # noqa: BLE001 - recorded on the row, not lost
        log.exception("run %s failed", run_id)
        try:
            with session_scope() as session:
                failed = session.get(Run, run_id)
                if failed is not None:
                    failed.status = "failed"
                    failed.error = str(exc)[:1000]
        except Exception:  # noqa: BLE001
            log.exception("could not record failure for run %s", run_id)


@router.get("", response_model=list[RunOut])
def list_runs(ontology_id: int | None = None, db: Session = Depends(get_db)):
    query = select(Run).order_by(Run.id.desc())
    if ontology_id is not None:
        query = query.where(Run.ontology_id == ontology_id)
    return [RunOut.of(run) for run in db.scalars(query)]


@router.post("/keys")
def preview_keys(payload: KeysIn):
    """Resolve a config to its stage keys without executing anything.

    Lets you see whether an edit will reuse retrieval before paying for it.
    """
    try:
        normalized = run_config.normalize(payload.config)
    except run_config.ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return run_config.stage_keys(normalized, payload.ontology_version)


@router.post("", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def start_run(payload: RunIn, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Register a run and start it in the background. Returns immediately."""
    try:
        run = runner.create_run(
            db,
            ontology_id=payload.ontology_id,
            config=payload.config,
            name=payload.name,
        )
    except run_config.ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    db.commit()
    background.add_task(
        _execute_in_background,
        run.id,
        document_limit=payload.document_limit,
        judge_limit=payload.judge_limit,
        skip_judge=payload.skip_judge,
    )
    return RunOut.of(run)


@router.get("/options")
def config_options():
    """What a run config can say, for a form: every default and every choice."""
    from hontology.judge import prompts
    from hontology.retrieve import embed

    defaults = run_config.normalize({})
    return {
        "defaults": {k: defaults[k] for k in ("common", "candidates", "judge")},
        "choices": {
            "selection": ["adaptive", "top-k"],
            "concept_fields": list(embed.CONCEPT_FIELD_SETS),
            "embed_providers": ["ollama", "llamacpp"],
            "judge_providers": ["ollama", "llamacpp", "openrouter"],
            "aggregation": list(run_config.AGGREGATIONS),
            "prompts": [
                {"prompt_id": prompt_id, "mode": prompts.get(prompt_id).mode}
                for prompt_id in prompts.available()
            ],
            "data_collection": ["deny", "allow"],
        },
    }


@router.get("/{run_id}/config")
def run_config_of(run_id: int, db: Session = Depends(get_db)):
    """A run's stored config, to start another from."""
    run = run_or_404(db, run_id)
    return run.config or {}


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Poll a run's status, stage and progress."""
    run = run_or_404(db, run_id)
    return RunOut.of(run)


@router.post("/{run_id}/resume", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def resume_run(
    run_id: int,
    background: BackgroundTasks,
    judge_limit: int | None = None,
    db: Session = Depends(get_db),
):
    """Continue a run, skipping pairs already judged."""
    run = run_or_404(db, run_id)

    background.add_task(
        _execute_in_background,
        run_id,
        document_limit=100,
        judge_limit=judge_limit,
        skip_judge=False,
    )
    return RunOut.of(run)


@router.get("/{run_id}/manifest")
def run_manifest(run_id: int, db: Session = Depends(get_db)):
    """The full provenance record: config, resolved keys, and the infrastructure used."""
    run = run_or_404(db, run_id)
    return run.manifest or {"config": run.config}
