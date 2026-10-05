"""Run orchestration: config in, candidates and verdicts out.

A `Run` row is both the experiment's identity and its job record, so a run that
dies leaves a row explaining how far it got rather than vanishing.

Reuse is decided here. Two runs sharing a ``candidates_key`` share retrieval, and
the second copies those rows instead of recomputing them — which is what makes
iterating on a prompt cost only the judging.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Document, Ontology, Run
from hontology.evalkit import config as run_config
from hontology.judge import run as judge_run_module
from hontology.ontology import snapshots
from hontology.retrieve import candidates as candidates_module

log = logging.getLogger(__name__)


def _infra_snapshot() -> dict:
    """Transport details, recorded for provenance and excluded from every key."""
    settings = get_settings()
    return {
        "ollama_host": settings.ollama_host,
        "llamacpp_host": settings.llamacpp_host,
        "llamacpp_embed_host": settings.llamacpp_embed_host or settings.llamacpp_host,
        "database": settings.psycopg_url().rsplit("@", 1)[-1],
    }


def create_run(
    session: Session,
    *,
    ontology_id: int,
    config: dict,
    name: str | None = None,
) -> Run:
    """Register a run, resolving its ontology version and stage keys."""
    # Validate the ontology up front. Without this, resolving a snapshot for a
    # nonexistent ontology inserts a row that fails on the foreign key, which
    # surfaces as an integrity error rather than "no such ontology".
    if session.get(Ontology, ontology_id) is None:
        raise LookupError(f"ontology {ontology_id} does not exist")

    normalized = run_config.normalize(config)
    if name:
        normalized["name"] = name

    version = snapshots.resolve(
        session, ontology_id, normalized["common"]["ontology_version"]
    ).version
    # Pin the concrete version: "latest" must not survive into the stored config,
    # or a replay months later would silently use different wording.
    normalized["common"]["ontology_version"] = version

    keys = run_config.stage_keys(normalized, version)
    run = Run(
        name=normalized["name"],
        ontology_id=ontology_id,
        ontology_version=version,
        config=normalized,
        manifest=run_config.manifest(normalized, version, infra=_infra_snapshot()),
        candidates_key=keys["candidates"],
        judge_key=keys["judge"],
        status="pending",
    )
    session.add(run)
    session.flush()
    return run


def reusable_candidates_run(session: Session, run: Run) -> Run | None:
    """A completed earlier run whose retrieval this one can adopt verbatim."""
    return session.scalar(
        select(Run)
        .where(
            Run.id != run.id,
            Run.ontology_id == run.ontology_id,
            Run.candidates_key == run.candidates_key,
            Run.status.in_(("done", "judged", "candidates")),
        )
        .order_by(Run.id)
        .limit(1)
    )


def copy_candidates(session: Session, source_run_id: int, target_run_id: int) -> int:
    rows = list(session.scalars(select(Candidate).where(Candidate.run_id == source_run_id)))
    for row in rows:
        session.add(
            Candidate(
                run_id=target_run_id,
                document_id=row.document_id,
                concept_id=row.concept_id,
                source=row.source,
                score=row.score,
                rank=row.rank,
                selected=row.selected,
                matched_code=row.matched_code,
                matched_level=row.matched_level,
            )
        )
    session.flush()
    return len(rows)


def pending_documents(session: Session, limit: int) -> list[Document]:
    """Documents with usable text, newest first."""
    return list(
        session.scalars(
            select(Document)
            .where(Document.body_path.is_not(None), Document.is_junk.is_(False))
            .order_by(Document.id.desc())
            .limit(limit)
        )
    )


def execute(
    session: Session,
    run: Run,
    *,
    document_limit: int = 200,
    judge_limit: int | None = None,
    skip_judge: bool = False,
    refresh_embeddings: bool = False,
) -> dict:
    """Run the pipeline for *run*, reusing retrieval where the key allows.

    Any failure — in either stage, including Ctrl-C — is recorded on the row
    before it propagates, with ``stage`` left on the stage that failed. A run
    left ``running`` by a crash is indistinguishable from one still in progress,
    and nothing would ever clear it.
    """
    run.status = "running"
    run.started_at = datetime.now(UTC)
    run.stage = "candidates"
    session.commit()

    try:
        result = _execute_stages(
            session,
            run,
            document_limit=document_limit,
            judge_limit=judge_limit,
            skip_judge=skip_judge,
            refresh_embeddings=refresh_embeddings,
        )
    except BaseException as exc:
        # The failed statement may have left the transaction unusable; the
        # stage's partial work is discarded, per-pair verdicts already
        # committed are kept for resume.
        session.rollback()
        run.status = "failed"
        run.error = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)[:1000]
        ) or type(exc).__name__
        run.finished_at = datetime.now(UTC)
        session.commit()
        raise

    run.stage = None
    run.finished_at = datetime.now(UTC)
    session.commit()
    return result


def _execute_stages(
    session: Session,
    run: Run,
    *,
    document_limit: int,
    judge_limit: int | None,
    skip_judge: bool,
    refresh_embeddings: bool,
) -> dict:
    documents = pending_documents(session, document_limit)
    reused_from = None

    existing = session.scalar(
        select(func.count(Candidate.id)).where(Candidate.run_id == run.id)
    )
    if existing:
        candidate_stats = {"reused": "already built for this run", "pool_rows": existing}
    else:
        donor = reusable_candidates_run(session, run)
        if donor is not None:
            copied = copy_candidates(session, donor.id, run.id)
            reused_from = donor.id
            candidate_stats = {"reused_from_run": donor.id, "pool_rows": copied}
            log.info("candidates reused from run %s (%s rows)", donor.id, copied)
        else:
            candidate_stats = candidates_module.build_candidates(
                session,
                run.id,
                ontology_id=run.ontology_id,
                documents=documents,
                config=run.config["candidates"],
                embed_body_limit=run.config["common"]["embed_body_limit"],
                refresh_embeddings=refresh_embeddings,
            )

    run.stage = "judge"
    session.commit()

    if skip_judge:
        # Terminal: retrieval only. The judge stage never started, so it must
        # not be left showing as the current stage.
        run.status = "candidates"
        return {
            "run_id": run.id,
            "candidates": candidate_stats,
            "reused_from_run": reused_from,
            "judge": None,
        }

    def progress(done: int, total: int) -> None:
        run.progress_done = done
        run.progress_total = total

    judge_stats = judge_run_module.judge_run(
        session,
        run.id,
        config=run.config,
        judge_body_limit=run.config["common"]["judge_body_limit"],
        limit=judge_limit,
        progress=progress,
    )
    run.status = "done"

    return {
        "run_id": run.id,
        "candidates": candidate_stats,
        "reused_from_run": reused_from,
        "judge": judge_stats,
        "liveness": judge_run_module.liveness(session, run.id),
    }
