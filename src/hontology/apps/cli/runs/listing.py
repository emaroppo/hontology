"""`hontology run` commands that inspect without executing: prompts, keys, runs."""

from __future__ import annotations

from pathlib import Path

import typer
from sqlalchemy import select

from hontology.apps.cli.common import read_json
from hontology.db.session import session_scope


def run_prompts() -> None:
    """List the registered prompt templates, with their fingerprints and pins."""
    from hontology.pipeline.judge import prompts
    from hontology.pipeline.runs import fingerprints

    for prompt_id in prompts.available():
        current = fingerprints.prompt_fingerprint(prompt_id)
        pinned = prompts.PINS.get(prompt_id)
        status = "pinned" if pinned == current else "UNPINNED" if pinned is None else "DRIFTED"
        typer.secho(
            f"{prompt_id:<20} {prompts.get(prompt_id).mode:<14} {current}  {status}",
            fg=None if status == "pinned" else typer.colors.YELLOW,
        )


def run_keys(config_path: Path, ontology_version: str = "v1") -> None:
    """Show the stage keys a config resolves to, without executing anything.

    Useful for checking what an edit will recompute before paying for it.
    """
    from hontology.pipeline.runs import config as run_config

    keys = run_config.stage_keys(run_config.normalize(read_json(config_path)), ontology_version)
    typer.echo(f"candidates  {keys['candidates']}")
    typer.echo(f"judge       {keys['judge']}")


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
