"""`hontology run` commands that execute a run: start, sample and resume."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import typer

from hontology.apps.cli.common import (
    After,
    Before,
    begin_run,
    calendar_documents,
    load_run,
    log_to_console,
    read_json,
    record_failure,
)
from hontology.db.session import session_scope

JudgeLimit = typer.Option(None, help="Cap pairs judged this pass.")


def run_start(
    ontology_id: int,
    config_path: Path,
    name: str | None = typer.Option(None, help="Override the config's name."),
    documents: int = typer.Option(100, help="How many documents to consider."),
    judge_limit: int | None = JudgeLimit,
    skip_judge: bool = typer.Option(False, help="Build candidates only."),
    refresh_embeddings: bool = typer.Option(
        False,
        help="Recompute document embeddings instead of reusing cached ones. For "
        "when the cache itself is the suspect; results are unaffected.",
    ),
    calendar_path: Path | None = typer.Option(
        None,
        "--calendar",
        help="Consider only fetched documents inside this calendar's windows, "
        "instead of the newest --documents.",
    ),
    before: Before = 1,
    after: After = 2,
) -> None:
    """Execute a run from a JSON config."""
    from hontology.pipeline.runs import runner

    log_to_console(logging.INFO)
    with session_scope() as session:
        run = runner.create_run(
            session, ontology_id=ontology_id, config=read_json(config_path), name=name
        )
        run_id = run.id
        typer.echo(f"run {run_id}: candidates={run.candidates_key} judge={run.judge_key}")

    document_ids = None
    if calendar_path is not None:
        with session_scope() as session:
            document_ids = sorted(calendar_documents(session, calendar_path, before, after))
            # Which documents a run considered is provenance, not behaviour: it
            # is recorded, never hashed, like the --documents limit.
            run = load_run(session, run_id)
            run.manifest = (run.manifest or {}) | {
                "documents": {
                    "calendar": str(calendar_path),
                    "window_days": [before, after],
                    "in_windows": len(document_ids),
                }
            }
        typer.echo(f"calendar: {len(document_ids)} document(s) in its windows")

    with session_scope() as session:
        result = runner.execute(
            session,
            load_run(session, run_id),
            document_limit=documents if document_ids is None else len(document_ids),
            judge_limit=judge_limit,
            skip_judge=skip_judge,
            refresh_embeddings=refresh_embeddings,
            document_ids=document_ids,
        )

    typer.echo(f"candidates  {result['candidates']}")
    if result.get("reused_from_run"):
        typer.secho(
            f"            retrieval reused from run {result['reused_from_run']}",
            fg=typer.colors.GREEN,
        )
    if result.get("judge"):
        typer.echo(f"judge       {result['judge']}")
        live = result["liveness"]
        typer.secho(
            f"liveness    {live['clean']}/{live['verdicts']} clean, {live['errors']} errors",
            fg=typer.colors.GREEN if live["ok"] else typer.colors.RED,
        )


def run_sample(
    ontology_id: int,
    config_path: Path,
    manifest_path: Path,
    candidates_from: int = typer.Option(
        ..., help="Reuse this run's retrieval for the sample's documents."
    ),
    first: int | None = typer.Option(
        None, help="Judge only the first N documents in the sample's frozen order."
    ),
    run_id: int | None = typer.Option(None, help="Extend this run instead of creating one."),
) -> None:
    """Judge a labelled sample's documents with another run's retrieval.

    An arm can be compared at article level as soon as documents are labelled,
    without judging every calendar window first. Rerun with a larger --first
    and the same --run-id as labelling continues; only new documents are judged.
    The run's calendar entries stay unprocessed until `run calendar` extends it.
    """
    from hontology.evaluation.calendar import runner as calendar_run

    log_to_console()
    manifest = read_json(manifest_path)
    order = [row["document_id"] for row in manifest["order"]]
    document_ids = order[:first] if first is not None else order

    with session_scope() as session:
        run = begin_run(session, run_id, ontology_id, config_path)
        # Same leaves, or the arms would be answering different questions.
        calendar_run.check_same_leaves(
            session, load_run(session, candidates_from), run.ontology_id
        )
        run.manifest = (
            {"calendar_done": []}
            | (run.manifest or {})
            | {
                "sample": {
                    "manifest": str(manifest_path),
                    "order_sha256": manifest["order_sha256"],
                    "judged_first": len(document_ids),
                    "candidates_from": candidates_from,
                }
            }
        )
        session.commit()
        typer.echo(f"run {run.id}: judging {len(document_ids)} sample document(s)")
        try:
            result = calendar_run.judge_documents(
                session, run, source_run_id=candidates_from, document_ids=document_ids
            )
        except BaseException as exc:
            # Verdicts already committed per pair are kept, for the next resume.
            session.rollback()
            record_failure(run, exc)
            session.commit()
            raise
        run.status = "done"
        run.finished_at = datetime.now(UTC)
        judge = result["judge"] or {}
        typer.echo(
            f"run {run.id}: {result['retrieved']} of {result['documents']} document(s) "
            f"retrieved by run {candidates_from}; judged {judge.get('judged', 0)}, "
            f"matched {judge.get('matched', 0)}, {judge.get('calls', 0)} call(s)"
        )


def run_resume(run_id: int, judge_limit: int | None = JudgeLimit) -> None:
    """Continue a run, skipping pairs already judged."""
    from hontology.pipeline.runs import runner

    log_to_console(logging.INFO)
    with session_scope() as session:
        result = runner.execute(session, load_run(session, run_id), judge_limit=judge_limit)
    typer.echo(f"judge  {result['judge']}")
