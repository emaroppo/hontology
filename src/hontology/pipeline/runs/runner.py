"""Run orchestration: config in, candidates and verdicts out.

A `Run` row is both the experiment's identity and its job record, so a run that
dies leaves a row explaining how far it got rather than vanishing.

Reuse is decided here: a run whose ``candidates_key`` another run already built
copies those rows (`pipeline.runs.reuse`) instead of recomputing them.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.models import Candidate, Document, Ontology, Run
from hontology.ontology import snapshots
from hontology.pipeline.judge import run as judge_run_module
from hontology.pipeline.retrieve import candidates as candidates_module
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs import fingerprints, reuse

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
    # Fingerprints of the code and prompt text in force now, so the run's stage
    # versions stay right if either is edited later.
    run.manifest = (run.manifest or {}) | {"versions": fingerprints.recorded(run)}
    session.add(run)
    session.flush()
    return run


def pending_documents(
    session: Session, limit: int, *, document_ids: list[int] | None = None
) -> list[Document]:
    """Documents with usable text, newest first, optionally within a given set.

    Near-duplicates are left out: only a group's representative is retrieved and
    judged, and its copies read its verdicts (see `pipeline.ingest.articles.dedup`).
    """
    query = select(Document).where(
        Document.body_path.is_not(None),
        Document.is_junk.is_(False),
        Document.duplicate_of.is_(None),
    )
    if document_ids is not None:
        query = query.where(among(Document.id, document_ids))
    return list(session.scalars(query.order_by(Document.id.desc()).limit(limit)))


def execute(
    session: Session,
    run: Run,
    *,
    document_limit: int = 200,
    judge_limit: int | None = None,
    skip_judge: bool = False,
    refresh_embeddings: bool = False,
    document_ids: list[int] | None = None,
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
            document_ids=document_ids,
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


def _candidates_stage(
    session: Session,
    run: Run,
    *,
    document_limit: int,
    refresh_embeddings: bool,
    document_ids: list[int] | None,
) -> tuple[dict, int | None]:
    """Build *run*'s candidates, or adopt another run's: the stats, and the donor."""
    documents = pending_documents(session, document_limit, document_ids=document_ids)
    existing = session.scalar(
        select(func.count(Candidate.id)).where(Candidate.run_id == run.id)
    )
    if existing:
        return {"reused": "already built for this run", "pool_rows": existing}, None
    donor = reuse.reusable_candidates_run(session, run)
    if donor is not None:
        copied = reuse.copy_candidates(session, donor.id, run.id)
        log.info("candidates reused from run %s (%s rows)", donor.id, copied)
        return {"reused_from_run": donor.id, "pool_rows": copied}, donor.id
    stats = candidates_module.build_candidates(
        session,
        run.id,
        ontology_id=run.ontology_id,
        documents=documents,
        config=run.config["candidates"],
        embed_body_limit=run.config["common"]["embed_body_limit"],
        refresh_embeddings=refresh_embeddings,
    )
    return stats, None


def _execute_stages(
    session: Session,
    run: Run,
    *,
    document_limit: int,
    judge_limit: int | None,
    skip_judge: bool,
    refresh_embeddings: bool,
    document_ids: list[int] | None,
) -> dict:
    candidate_stats, reused_from = _candidates_stage(
        session,
        run,
        document_limit=document_limit,
        refresh_embeddings=refresh_embeddings,
        document_ids=document_ids,
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
