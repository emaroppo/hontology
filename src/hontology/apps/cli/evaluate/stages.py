"""`hontology eval` on pipeline stages: the pre-scrape filter and parameter sweeps."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from hontology.apps.cli.common import log_to_console, read_json
from hontology.db.session import session_scope


def eval_filter_report(ontology_id: int) -> None:
    """Per-code cost and benefit of the pre-scrape filter mapping."""
    from hontology.evaluation.stages import filter_report

    with session_scope() as session:
        typer.echo(filter_report.format_report(filter_report.report(session, ontology_id)))


def eval_sweep(
    ontology_id: int,
    sweep_path: Path,
    execute: bool = typer.Option(False, help="Run the plan, not just print it."),
    documents: int = typer.Option(50, help="Documents per cell."),
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged per cell."),
) -> None:
    """Plan or run a parameter sweep. Cells already run are skipped."""
    from hontology.pipeline.runs import sweep as sweep_module

    log_to_console(logging.INFO)
    spec = read_json(sweep_path)

    def plan(session):
        return sweep_module.plan(
            session,
            ontology_id,
            base=spec.get("base", {}),
            axes=spec.get("axes", {}),
            name_prefix=spec.get("name", "sweep"),
        )

    with session_scope() as session:
        summary = plan(session).as_dict()

    typer.echo(
        f"{summary['total']} cell(s): {summary['done']} already run, "
        f"{summary['todo']} to do, {summary['distinct_candidate_keys']} distinct "
        f"retrieval key(s)"
    )
    for cell in summary["cells"]:
        mark = "done" if cell["done"] else "todo"
        typer.echo(f"  [{mark}] {cell['name']}  cand={cell['candidates_key']}")

    if not execute:
        typer.secho("\nplan only — pass --execute to run", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    with session_scope() as session:
        results = sweep_module.execute(
            session,
            ontology_id,
            plan(session),
            document_limit=documents,
            judge_limit=judge_limit,
        )
    typer.secho(f"\nran {len(results)} cell(s)", fg=typer.colors.GREEN)
