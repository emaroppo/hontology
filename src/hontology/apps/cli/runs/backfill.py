"""`hontology run backfill-shown`: rebuild the shown record of older runs."""

from __future__ import annotations

import typer

from hontology.db.session import session_scope


def run_backfill_shown(
    run_ids: list[int] = typer.Argument(
        None, help="Runs to backfill; all unrecorded ones if none."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be written."),
) -> None:
    """Rebuild what runs made before the shown record showed their judge.

    Read from their verdicts, one call per transaction (see
    `pipeline.runs.backfill`). A run that already has a shown record is left alone.
    """
    from hontology.db.models import Run
    from hontology.pipeline.runs import backfill

    with session_scope() as session:
        unrecorded = {run.id: run for run in backfill.unrecorded_runs(session)}
        chosen = run_ids or sorted(unrecorded)
        for run_id in chosen:
            run = unrecorded.get(run_id)
            if run is None:
                state = (
                    "no such run" if session.get(Run, run_id) is None else "already recorded"
                )
                typer.echo(f"run {run_id}: skipped, {state}")
                continue
            result = backfill.backfill_run(session, run, write=not dry_run)
            skipped = ", ".join(f"{n} {why}" for why, n in result.not_shown.items()) or "none"
            typer.echo(
                f"run {run_id} ({result.strategy}): {result.articles} articles, "
                f"{result.offers} calls, {result.shown} shown pairs; "
                f"leaf rows not shown: {skipped}"
            )
        if dry_run:
            session.rollback()
            typer.echo("dry run: nothing written")
