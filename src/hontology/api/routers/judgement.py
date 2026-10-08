"""Judgement trials: what the model says to a pair under a prompt of choice.

Asked exactly as the chosen run asks (provider, model, decoding, article text
limit), but nothing is recorded: see `judge.trial`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Run, Verdict
from hontology.db.session import get_db
from hontology.evalkit import config as run_config
from hontology.evalkit import judge_eval
from hontology.judge import trial
from hontology.retrieve import tuning

router = APIRouter(prefix="/judgement", tags=["judgement"])


class TrialIn(BaseModel):
    run_id: int
    document_id: int
    concept_id: int
    prompt_id: str = "strict_v1"
    system: str | None = None
    # Any of definition / inclusion_criteria / exclusion_criteria; not saved.
    wording: dict[str, str | None] | None = None


class ArticlesIn(BaseModel):
    run_id: int
    concept_id: int
    annotator: str | None = None
    limit: int = 40


def _trial(payload: TrialIn) -> trial.Trial:
    return trial.Trial(**payload.model_dump())


def _refuse(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))


@router.get("/templates")
def templates():
    """Per-pair prompts, the ones a trial can use, with their system text."""
    return trial.per_pair_templates()


@router.get("/runs")
def runs(ontology_id: int, db: Session = Depends(get_db)):
    """Runs of an ontology with verdicts, and the model each asks."""
    judged: dict[int, int] = {
        run_id: n
        for run_id, n in db.execute(
            select(Verdict.run_id, func.count())
            .join(Run, Run.id == Verdict.run_id)
            .where(Run.ontology_id == ontology_id)
            .group_by(Verdict.run_id)
        )
    }
    out = []
    for run in db.scalars(select(Run).where(Run.id.in_(list(judged))).order_by(Run.id.desc())):
        judge = run_config.normalize(run.config or {})["judge"]
        out.append(
            {
                "id": run.id,
                "name": run.name,
                "verdicts": judged[run.id],
                "provider": judge["provider"],
                "model": judge["model"],
                "prompt_id": judge["prompt_id"],
            }
        )
    return out


@router.post("/articles")
def articles(payload: ArticlesIn, db: Session = Depends(get_db)):
    run = db.get(Run, payload.run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {payload.run_id} does not exist")
    try:
        labels = tuning.truth(db, run.ontology_id, payload.annotator)
    except LookupError as exc:
        raise _refuse(exc) from exc
    return trial.articles(db, payload.run_id, payload.concept_id, labels, limit=payload.limit)


@router.post("/render")
def render(payload: TrialIn, db: Session = Depends(get_db)):
    """The prompt a trial would send, without sending it."""
    try:
        return trial.render(db, _trial(payload))
    except (LookupError, ValueError) as exc:
        raise _refuse(exc) from exc


@router.post("/ask")
def ask(payload: TrialIn, db: Session = Depends(get_db)):
    """Send a trial to the model. A provider failure comes back in ``error``."""
    try:
        return trial.ask(db, _trial(payload))
    except (LookupError, ValueError) as exc:
        raise _refuse(exc) from exc


# --- Judge versions, scored on what they were given --------------------------


@router.get("/versions")
def versions(ontology_id: int, db: Session = Depends(get_db)):
    return judge_eval.list_versions(db, ontology_id)


class VersionEvalIn(BaseModel):
    run_ids: list[int]
    scope: str = "responsible"
    annotator: str | None = None


@router.post("/versions/evaluate")
def evaluate_version(payload: VersionEvalIn, db: Session = Depends(get_db)):
    """A judge version's precision and recall, per run and pooled, on the pairs
    it was responsible for or only on those retrieval selected."""
    if not payload.run_ids:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "no runs given")
    run = db.get(Run, payload.run_ids[0])
    if run is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"run {payload.run_ids[0]} does not exist"
        )
    try:
        labels = tuning.truth(db, run.ontology_id, payload.annotator)
        return judge_eval.evaluate(db, payload.run_ids, labels, scope=payload.scope)
    except (LookupError, ValueError) as exc:
        raise _refuse(exc) from exc
