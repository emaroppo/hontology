"""Options and helpers the command groups share."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

Before = Annotated[int, typer.Option(help="Calendar window: days of feed before each date.")]
After = Annotated[int, typer.Option(help="Calendar window: days of feed after each date.")]
IncludeMachine = Annotated[bool, typer.Option(help="Count machine labels.")]


def log_to_console(level: int = logging.WARNING) -> None:
    logging.basicConfig(level=level, format="%(levelname)-5s %(message)s")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(text: str, out: Path | None, *, count_rows: bool = False) -> None:
    """Print `text`, or write it to `out` and say so."""
    if out is None:
        typer.echo(text)
        return
    out.write_text(text, encoding="utf-8")
    rows = f" ({len(text.splitlines()) - 1} row(s))" if count_rows else ""
    typer.echo(f"wrote {out}{rows}")


def write_csv(out: Path, columns: Iterable[str], rows: list[dict]) -> None:
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def calendar_documents(session, calendar_path: Path, before: int, after: int) -> set[int]:
    """Every document inside a calendar's windows."""
    from hontology.evaluation.calendar import events

    return events.all_window_documents(
        session, events.load(calendar_path), before=before, after=after
    )


def load_run(session, run_id: int):
    """Fetch a run or exit with a clear message rather than a traceback."""
    from hontology.db.models import Run

    run = session.get(Run, run_id)
    if run is None:
        typer.secho(f"no run with id {run_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    return run


def begin_run(session, run_id: int | None, ontology_id: int, config_path: Path):
    """Create a run from a config, or reopen `run_id`, and mark it running."""
    from hontology.pipeline.runs import runner

    if run_id is None:
        run = runner.create_run(session, ontology_id=ontology_id, config=read_json(config_path))
    else:
        run = load_run(session, run_id)
    run.status = "running"
    run.started_at = run.started_at or datetime.now(UTC)
    # A resumed run must not keep the finish time or error of its last pass.
    run.finished_at = None
    run.error = None
    return run


def record_failure(run, exc: BaseException) -> None:
    run.status = "failed"
    run.error = (
        "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)[:1000]
    ) or type(exc).__name__
    run.finished_at = datetime.now(UTC)
