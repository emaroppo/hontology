"""Retrieval exploration: a finished run's ranked pool under a cutoff of choice.

Nothing here embeds anything. Every run stores its whole pool before the cutoff,
so these endpoints only draw a different line through a ranking that already
exists (see `retrieve.tuning`). Requests are POSTs because each carries the
cutoff and, optionally, a machine annotation set to score against instead of
the human labels.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Run
from hontology.db.session import get_db
from hontology.evalkit import versions
from hontology.retrieve import live, tuning

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class CutoffIn(BaseModel):
    selection: str = Field("adaptive", pattern="^(adaptive|top-k)$")
    top_k: int = Field(3, ge=1)
    min_score: float = Field(0.45, ge=-1, le=1)
    rel_margin: float = Field(0.05, ge=0, le=2)
    max_k: int = Field(8, ge=1)


class ExploreIn(BaseModel):
    # None is the cutoff the run was built with.
    cutoff: CutoffIn | None = None
    # The truth: None for human labels, else a machine annotation set's name.
    annotator: str | None = None


def _run(db: Session, run_id: int) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id} does not exist")
    return run


def _cutoff(run: Run, payload: ExploreIn) -> tuning.Cutoff:
    if payload.cutoff is None:
        return tuning.run_cutoff(run)
    return tuning.Cutoff(**payload.cutoff.model_dump())


def _truth(db: Session, run: Run, payload: ExploreIn) -> tuning.Truth:
    try:
        return tuning.truth(db, run.ontology_id, payload.annotator)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/runs")
def list_runs(ontology_id: int, db: Session = Depends(get_db)):
    """Runs of an ontology that have a stored pool, newest first."""
    counts: dict[int, int] = {
        run_id: n
        for run_id, n in db.execute(
            select(Candidate.run_id, func.count(func.distinct(Candidate.document_id)))
            .join(Run, Run.id == Candidate.run_id)
            .where(Run.ontology_id == ontology_id)
            .group_by(Candidate.run_id)
        )
    }
    runs = db.scalars(select(Run).where(Run.id.in_(list(counts))).order_by(Run.id.desc()))
    return [
        {
            "id": run.id,
            "name": run.name,
            "status": run.status,
            "documents": counts[run.id],
            "cutoff": asdict(tuning.run_cutoff(run)),
        }
        for run in runs
    ]


@router.post("/runs/{run_id}/report")
def cutoff_report(run_id: int, payload: ExploreIn, db: Session = Depends(get_db)):
    """What a cutoff keeps and what it costs in recall, beside the run's own."""
    run = _run(db, run_id)
    labels = _truth(db, run, payload)
    setting, own = _cutoff(run, payload), tuning.run_cutoff(run)
    report = tuning.report(db, run_id, own, labels)
    # The run's own cutoff is computed once when it is also the one asked about.
    return {
        "setting": report if setting == own else tuning.report(db, run_id, setting, labels),
        "run": report,
    }


@router.post("/runs/{run_id}/labelled")
def labelled_documents(run_id: int, payload: ExploreIn, db: Session = Depends(get_db)):
    run = _run(db, run_id)
    return tuning.labelled_documents(db, run_id, _truth(db, run, payload))


@router.post("/runs/{run_id}/documents/{document_id}")
def document_pool(
    run_id: int, document_id: int, payload: ExploreIn, db: Session = Depends(get_db)
):
    run = _run(db, run_id)
    try:
        return tuning.document_pool(
            db, run_id, document_id, _cutoff(run, payload), _truth(db, run, payload)
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/runs/{run_id}/concepts/{concept_id}")
def concept_documents(
    run_id: int,
    concept_id: int,
    payload: ExploreIn,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    run = _run(db, run_id)
    return tuning.concept_documents(
        db, run_id, concept_id, _cutoff(run, payload), _truth(db, run, payload), limit=limit
    )


# --- Retrieval versions, scored live ------------------------------------------


@router.get("/versions")
def list_versions(ontology_id: int, db: Session = Depends(get_db)):
    """Retrieval versions of an ontology's runs, newest first, each with the runs
    that used it and their cutoffs as presets."""
    grouped: dict[str, dict] = {}
    for run in db.scalars(
        select(Run).where(Run.ontology_id == ontology_id).order_by(Run.id.desc())
    ):
        version = versions.retrieval_version(db, run)
        entry = grouped.setdefault(version["version"], version | {"runs": []})
        source = db.get(Run, version["source_run"])
        entry["runs"].append(
            {
                "id": run.id,
                "name": run.name,
                "cutoff": asdict(tuning.run_cutoff(source or run)),
            }
        )
    return list(grouped.values())


class LiveIn(BaseModel):
    run_id: int
    cutoff: CutoffIn
    pool_size: int = Field(20, ge=1, le=100)
    annotator: str | None = None


@router.post("/versions/evaluate")
def evaluate_version(payload: LiveIn, db: Session = Depends(get_db)):
    """A retrieval version (that of *run_id*) scored live on every labelled
    article under *cutoff*. Nothing is embedded or stored."""
    run = _run(db, payload.run_id)
    labels = _truth(db, run, ExploreIn(annotator=payload.annotator))
    return live.live_report(
        db,
        run,
        tuning.Cutoff(**payload.cutoff.model_dump()),
        labels,
        pool_size=payload.pool_size,
    )


# --- Pasted text, ranked as a run would rank an article -----------------------


class TextIn(BaseModel):
    run_id: int
    text: str = Field(min_length=1)
    cutoff: CutoffIn | None = None
    pool_size: int = Field(20, ge=1, le=100)


@router.post("/text")
def rank_text(payload: TextIn, db: Session = Depends(get_db)):
    """Rank the leaves against pasted text with *run_id*'s retrieval and cut it.
    The text is embedded and dropped; nothing about it is stored."""
    from hontology.judge.providers.base import ProviderError
    from hontology.retrieve import live

    run = _run(db, payload.run_id)
    cutoff = (
        tuning.Cutoff(**payload.cutoff.model_dump())
        if payload.cutoff is not None
        else tuning.run_cutoff(versions.source_run(db, run))
    )
    try:
        return live.rank_text(db, run, payload.text, cutoff, pool_size=payload.pool_size)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"embedding model unavailable: {exc}"
        ) from exc
