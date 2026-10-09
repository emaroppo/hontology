"""`hontology run`: configure and execute detection runs."""

from __future__ import annotations

import logging
import signal
from datetime import UTC, datetime
from pathlib import Path

import typer
from sqlalchemy import select

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
from hontology.pipeline.ingest.feed import slices

app = typer.Typer(help="Configure and execute detection runs.")
JudgeLimit = typer.Option(None, help="Cap pairs judged this pass.")


def _sigterm_as_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


@app.callback()
def run_signals() -> None:
    """Configure and execute detection runs."""
    # A run stopped with SIGTERM (`kill`, a process manager) would otherwise end
    # without raising anything, skipping the handlers that record an interrupted
    # run on its row, and leave it "running" forever. Raised as Ctrl-C is, it is
    # recorded the same way.
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)


@app.command("prompts")
def run_prompts() -> None:
    """List the registered prompt templates, with their fingerprints and pins."""
    from hontology.pipeline.judge import prompts
    from hontology.pipeline.runs import versions

    for prompt_id in prompts.available():
        current = versions.prompt_fingerprint(prompt_id)
        pinned = prompts.PINS.get(prompt_id)
        status = "pinned" if pinned == current else "UNPINNED" if pinned is None else "DRIFTED"
        typer.secho(
            f"{prompt_id:<20} {prompts.get(prompt_id).mode:<14} {current}  {status}",
            fg=None if status == "pinned" else typer.colors.YELLOW,
        )


@app.command("keys")
def run_keys(config_path: Path, ontology_version: str = "v1") -> None:
    """Show the stage keys a config resolves to, without executing anything.

    Useful for checking what an edit will recompute before paying for it.
    """
    from hontology.pipeline.runs import config as run_config

    keys = run_config.stage_keys(run_config.normalize(read_json(config_path)), ontology_version)
    typer.echo(f"candidates  {keys['candidates']}")
    typer.echo(f"judge       {keys['judge']}")


@app.command("list")
def run_list() -> None:
    """List runs with their status and stage keys."""
    from hontology.db.models import Run

    with session_scope() as session:
        runs = list(session.scalars(select(Run).order_by(Run.id)))
        if not runs:
            typer.echo("(no runs yet)")
            return
        for run in runs:
            typer.echo(
                f"{run.id:>4}  {run.status:<10} {run.name:<24} "
                f"{run.ontology_version:<5} cand={run.candidates_key} "
                f"judge={run.judge_key.split('_')[0]}"
            )


@app.command("start")
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


@app.command("sample")
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


def _window_ready(entry, places, before: int, after: int) -> bool:
    """Whether a window's feed is all in, re-ingesting its failed slices once."""
    from hontology.evaluation.calendar import runner as calendar_run

    with session_scope() as session:
        ready, failed = calendar_run.window_status(
            session, entry, places, before=before, after=after
        )
    if ready or not failed:
        return ready
    for feed, key in failed:
        with session_scope() as session:
            scope = places if feed == slices.FEED_GKG else None
            slices.ingest_slice(session, key, feed=feed, loci=scope)
    with session_scope() as session:
        ready, _ = calendar_run.window_status(
            session, entry, places, before=before, after=after
        )
    return ready


@app.command("calendar")
def run_calendar(
    ontology_id: int,
    config_path: Path,
    calendar_path: Path,
    run_id: int | None = typer.Option(None, help="Resume this run instead of creating one."),
    budget: int | None = typer.Option(
        None, help="Pairs judged per window, highest retrieval score first. Default: all."
    ),
    before: Before = 1,
    after: After = 2,
    poll: int = typer.Option(120, help="Seconds to wait when no window is ready yet."),
    prepare_only: bool = typer.Option(
        False,
        help="Scrape, deduplicate and retrieve, but judge nothing and mark nothing "
        "finished, to see each window's judging volume first.",
    ),
    candidates_from: int | None = typer.Option(
        None,
        help="Reuse this run's retrieval: judge exactly its documents in each window, "
        "once it has finished that window. For comparing judging arms.",
    ),
) -> None:
    """Run a calendar window by window, as soon as each window's feed is ingested.

    Each ready window is scraped, deduplicated, retrieved and judged into one
    run; finished entries are recorded on the run, so a restart with --run-id
    continues where it stopped. Score it afterwards with `eval calendar`.
    """
    import time

    from hontology.evaluation.calendar import events
    from hontology.evaluation.calendar import runner as calendar_run

    log_to_console()
    entries = events.load(calendar_path)

    with session_scope() as session:
        loci = events.loci_for(session, entries)
        run = begin_run(session, run_id, ontology_id, config_path)
        run.stage = "calendar"
        run.manifest = (run.manifest or {}) | {
            "documents": {
                "calendar": str(calendar_path),
                "window_days": [before, after],
                "budget_per_window": budget,
                "candidates_from": candidates_from,
            }
        }
        if candidates_from is not None:
            # Same leaves, or the arms would be answering different questions.
            calendar_run.check_same_leaves(
                session, load_run(session, candidates_from), run.ontology_id
            )
        run_id = run.id
        # A prepare-only pass tracks its own progress, so a later judging pass
        # still visits every window.
        progress_key = "calendar_prepared" if prepare_only else "calendar_done"
        finished = set((run.manifest or {}).get(progress_key, []))
        # Windows the source published nothing for: set aside, never scored.
        unobservable = set((run.manifest or {}).get("calendar_unobservable", []))
    typer.echo(f"run {run_id}: {len(finished)}/{len(entries)} entries already done")
    unobservable &= {e.id for e in entries}
    if unobservable:
        typer.echo(f"  set aside, no source data: {', '.join(sorted(unobservable))}")

    try:
        while len(finished | unobservable) < len(entries):
            progressed = False
            for entry in entries:
                if entry.id in finished or entry.id in unobservable:
                    continue
                places = [loci[c] for c in entry.countries]
                ready = _window_ready(entry, places, before, after)
                if ready:
                    with session_scope() as session:
                        if calendar_run.window_unobservable(
                            session, entry, places, before=before, after=after
                        ):
                            unobservable.add(entry.id)
                            run = load_run(session, run_id)
                            stored = set((run.manifest or {}).get("calendar_unobservable", []))
                            run.manifest = (run.manifest or {}) | {
                                "calendar_unobservable": sorted(stored | {entry.id})
                            }
                            typer.echo(
                                f"[set aside] {entry.id}: the source published nothing "
                                "for this window; not scored"
                            )
                            progressed = True
                            continue
                if ready and candidates_from is not None:
                    # Reusing another run's retrieval: wait until it has
                    # finished this window, so the documents are all there.
                    with session_scope() as session:
                        source = load_run(session, candidates_from)
                        ready = entry.id in (source.manifest or {}).get("calendar_done", [])
                if not ready:
                    continue
                with session_scope() as session:
                    run = load_run(session, run_id)
                    summary = calendar_run.process_entry(
                        session,
                        run,
                        entry,
                        places,
                        before=before,
                        after=after,
                        budget=budget,
                        judge=not prepare_only,
                        candidates_from=candidates_from,
                    )
                    finished.add(entry.id)
                    run.manifest = (run.manifest or {}) | {progress_key: sorted(finished)}
                judge = summary["judge"] or {}
                judge_note = (
                    f"{summary['pairs_selected']} pair(s) selected"
                    if prepare_only
                    else f"judged {judge.get('judged', 0)}, matched {judge.get('matched', 0)}"
                )
                typer.echo(
                    f"[{len(finished)}/{len(entries)}] {entry.id}: "
                    f"{summary['passed_filter']} in scope, "
                    f"{summary['representatives']} unique, {judge_note}"
                )
                progressed = True
            if not progressed and len(finished | unobservable) < len(entries):
                time.sleep(poll)
    except BaseException as exc:
        with session_scope() as session:
            record_failure(load_run(session, run_id), exc)
        raise

    with session_scope() as session:
        run = load_run(session, run_id)
        if prepare_only:
            # Prepared, not judged: left resumable rather than marked done.
            run.status = "candidates"
            run.stage = "judge"
            typer.echo(f"run {run_id} prepared; judge it with --run-id {run_id}")
            return
        run.status = "done"
        run.stage = None
        run.finished_at = datetime.now(UTC)
    typer.echo(
        f"run {run_id} done; score it with: hontology eval calendar {run_id} {calendar_path}"
    )


@app.command("resume")
def run_resume(run_id: int, judge_limit: int | None = JudgeLimit) -> None:
    """Continue a run, skipping pairs already judged."""
    from hontology.pipeline.runs import runner

    log_to_console(logging.INFO)
    with session_scope() as session:
        result = runner.execute(session, load_run(session, run_id), judge_limit=judge_limit)
    typer.echo(f"judge  {result['judge']}")
