"""`hontology eval`: score runs against the ground-truth bank.

Commands live in one module per topic and are registered here, in one table,
so `eval --help` lists them in a deliberate order rather than in import order.
"""

from __future__ import annotations

import typer

from hontology.apps.cli.evaluate import (
    arms,
    calendar,
    outputs,
    pairs,
    regression,
    review,
    stages,
)

app = typer.Typer(help="Score runs against the ground-truth bank.")

for name, command in (
    ("run", pairs.eval_run),
    ("breakdown", pairs.eval_breakdown),
    ("errors", pairs.eval_errors),
    ("filter-report", stages.eval_filter_report),
    ("sweep", stages.eval_sweep),
    ("funnel", outputs.eval_funnel),
    ("calendar", calendar.eval_calendar),
    ("arms", arms.eval_arms),
    ("calendar-review-export", review.eval_calendar_review_export),
    ("calendar-review-import", review.eval_calendar_review_import),
    ("detections", outputs.eval_detections),
    ("compare", pairs.eval_compare),
    ("consistency", pairs.eval_consistency),
    ("baseline", regression.eval_baseline),
    ("gate", regression.eval_gate),
    ("leaderboard", regression.eval_leaderboard),
):
    app.command(name)(command)
