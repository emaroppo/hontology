"""`hontology run`: configure and execute detection runs.

Commands live in one module per topic and are registered here, in one table,
so `run --help` lists them in a deliberate order rather than in import order.
"""

from __future__ import annotations

import signal

import typer

from hontology.apps.cli.runs import calendar, execute, listing

app = typer.Typer(help="Configure and execute detection runs.")


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


for name, command in (
    ("prompts", listing.run_prompts),
    ("keys", listing.run_keys),
    ("list", listing.run_list),
    ("start", execute.run_start),
    ("sample", execute.run_sample),
    ("calendar", calendar.run_calendar),
    ("resume", execute.run_resume),
):
    app.command(name)(command)
