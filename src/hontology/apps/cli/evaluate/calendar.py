"""`hontology eval calendar`: event-level results against a calendar of known events."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hontology.apps.cli.common import After, Before
from hontology.db.session import session_scope

# The calendar's headline rates, in the order reports show them.
CALENDAR_RATES = (
    ("event recall", "event_recall"),
    ("precursor recall", "precursor_recall"),
    ("false alarm rate", "false_alarm_rate"),
)


def rate_text(stat: dict, sep: str = " ") -> str:
    value = "-" if stat["rate"] is None else f"{stat['rate']:.2f}"
    return f"{stat['hits']}/{stat['n']}{sep}{value}"


def eval_calendar(
    run_id: int,
    calendar_path: Path,
    before: Before = 1,
    after: After = 2,
    out: Path | None = typer.Option(None, help="Also write the full result as JSON."),
) -> None:
    """Event-level results against a calendar of known events. Needs no labels."""
    from hontology.evaluation.calendar import events as calendar
    from hontology.evaluation.calendar import score as calendar_score

    with session_scope() as session:
        result = calendar_score.evaluate(
            session, run_id, calendar.load(calendar_path), before=before, after=after
        )
    if out is not None:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_entries(result)
    typer.echo("")
    _print_summary(result)


def _print_entries(result: dict) -> None:
    """One line per calendar entry: its volume at each stage, and the outcome."""
    from hontology.evaluation.calendar import events as calendar

    # `unique` sits after `fetched`: what is left once near-duplicates collapse.
    columns = (*calendar.STAGES[:3], "unique", *calendar.STAGES[3:])
    typer.echo(
        f"{'entry':34} {'kind':9} " + " ".join(f"{c[:8]:>8}" for c in columns) + "  result"
    )
    for row in result["entries"]:
        control = row["kind"] == calendar.CONTROL
        if not row["detected"]:
            verdict = "quiet" if control else f"lost at {row['lost_at']}"
        elif control and row["verified"] is True:
            verdict = "control withdrawn: a real instance was found"
        else:
            # A person's review of the matches: confirmed, rejected, or not yet.
            review = {True: "confirmed", False: "rejected", None: "unreviewed"}[row["verified"]]
            verdict = f"{'FALSE ALARM' if control else 'detected'} ({review})"
        typer.echo(
            f"{row['id'][:34]:34} {row['kind']:9} "
            + " ".join(f"{row[c]:>8}" for c in columns)
            + f"  {verdict}"
        )


def _print_summary(result: dict) -> None:
    """The headline rates, raw and verified, then cost, gaps and lead times."""
    from hontology.evaluation.metrics.intervals import format_ci

    verified = result["summary"]["verified"]

    def rates(stats: dict, indent: str) -> None:
        for label, key in CALENDAR_RATES:
            stat = stats[key]
            typer.echo(f"{indent + label:18} {rate_text(stat, '  ')}  {format_ci(*stat['ci'])}")

    rates(result["summary"], "")
    typer.echo(f"positives lost at  {result['summary']['positives_lost_at']}")
    typer.echo("verified by review:")
    rates(verified, "  ")
    if verified["controls_withdrawn"]:
        typer.echo(f"  withdrawn        {', '.join(verified['controls_withdrawn'])}")
    if verified["pending_review"]:
        pending = verified["pending_review"]
        typer.echo(f"  pending review   {len(pending)}: {', '.join(pending)}")
    cost = result["cost"]
    typer.echo(
        f"judging cost       {cost['pairs']} pair(s), {cost['input_tokens']} in / "
        f"{cost['output_tokens']} out tokens, {cost['seconds'] / 3600:.1f} h"
    )
    if result["not_processed"]:
        skipped = result["not_processed"]
        typer.echo(f"not processed      {len(skipped)}, not scored: {', '.join(skipped)}")
    for lead in result["lead_times"]:
        typer.echo(
            f"lead  {lead['precursor']} -> {lead['disruption']}: {lead['lead_days']} day(s)"
            f"{'' if lead['disruption_detected'] else '  (disruption itself missed)'}"
        )
