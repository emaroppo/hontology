"""`hontology run calendar`: run a calendar window by window as its feed arrives."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from hontology.apps.cli.common import (
    After,
    Before,
    begin_run,
    load_run,
    log_to_console,
    record_failure,
)
from hontology.apps.cli.runs.windows import CalendarWalk
from hontology.db.session import session_scope


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

    log_to_console()
    entries = events.load(calendar_path)
    walk = _open(
        entries,
        run_id,
        ontology_id,
        config_path,
        calendar_path,
        before=before,
        after=after,
        budget=budget,
        prepare_only=prepare_only,
        candidates_from=candidates_from,
    )
    typer.echo(f"run {walk.run_id}: {len(walk.finished)}/{len(entries)} entries already done")
    walk.unobservable &= {e.id for e in entries}
    if walk.unobservable:
        typer.echo(f"  set aside, no source data: {', '.join(sorted(walk.unobservable))}")

    try:
        while walk.pending():
            progressed = False
            for entry in entries:
                if entry.id in walk.finished or entry.id in walk.unobservable:
                    continue
                progressed = walk.visit(entry) or progressed
            if not progressed and walk.pending():
                time.sleep(poll)
    except BaseException as exc:
        with session_scope() as session:
            record_failure(load_run(session, walk.run_id), exc)
        raise

    _close(walk.run_id, prepare_only, calendar_path)


def _open(
    entries: list,
    run_id: int | None,
    ontology_id: int,
    config_path: Path,
    calendar_path: Path,
    *,
    before: int,
    after: int,
    budget: int | None,
    prepare_only: bool,
    candidates_from: int | None,
) -> CalendarWalk:
    """Create or reopen the run, record the calendar on it, and read its progress."""
    from hontology.evaluation.calendar import events
    from hontology.evaluation.calendar import runner as calendar_run

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
        walk = CalendarWalk(
            run_id=run.id,
            entries=entries,
            loci=loci,
            before=before,
            after=after,
            budget=budget,
            prepare_only=prepare_only,
            candidates_from=candidates_from,
            finished=set(),
            unobservable=set(),
        )
        manifest = run.manifest or {}
        walk.finished = set(manifest.get(walk.progress_key, []))
        walk.unobservable = set(manifest.get("calendar_unobservable", []))
    return walk


def _close(run_id: int, prepare_only: bool, calendar_path: Path) -> None:
    """Mark the run done, or, after a prepare-only pass, ready to be judged."""
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
