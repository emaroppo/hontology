"""`hontology eval arms`: the pre-registered comparison of arms against a baseline."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hontology.apps.cli.common import After, Before
from hontology.apps.cli.evaluate.calendar import rate_text
from hontology.db.session import session_scope


def eval_arms(
    baseline: int,
    calendar_path: Path,
    arm: list[int] = typer.Option([], "--arm", help="An arm's run id; repeat for several."),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Labelled-sample manifest."
    ),
    out: Path | None = typer.Option(None, help="Write the full report as JSON."),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Write the report as Markdown tables, for the write-up."
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        help="Score against this whole-document labels CSV instead of the label bank.",
    ),
    annotator: str | None = typer.Option(
        None,
        "--annotator",
        help="Score against this machine annotation set instead of the label bank.",
    ),
    before: Before = 1,
    after: After = 2,
) -> None:
    """The pre-registered comparison: every arm against the baseline, both levels."""
    from hontology.evaluation.calendar import events as calendar
    from hontology.evaluation.comparison import arms, arms_report

    if labels_path is not None and annotator is not None:
        raise typer.BadParameter("give --labels or --annotator, not both")
    with session_scope() as session:
        labels = _truth(session, baseline, labels_path, annotator)
        report = arms.compare_arms(
            session,
            baseline,
            arm,
            calendar.load(calendar_path),
            arms.load_manifest(manifest_path),
            before=before,
            after=after,
            labels=labels,
        )
    if out is not None:
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if markdown is not None:
        markdown.write_text(arms_report.render_markdown(report), encoding="utf-8")
    _print_runs(report, baseline)
    _print_sample(report.get("sample"))
    _print_comparisons(report)


def _truth(session, baseline: int, labels_path: Path | None, annotator: str | None):
    """The labels to score against: None for the label bank, else the CSV or
    machine annotation set asked for, read for the baseline run's ontology."""
    from hontology.db.models import Run
    from hontology.evaluation.labels import annotations
    from hontology.evaluation.labels.document_labels import document_label_map

    if labels_path is None and annotator is None:
        return None
    baseline_run = session.get(Run, baseline)
    if baseline_run is None:
        raise typer.BadParameter(f"run {baseline} does not exist")
    if labels_path is not None:
        return document_label_map(
            session, baseline_run.ontology_id, labels_path.read_text(encoding="utf-8")
        )
    try:
        return annotations.truth(session, baseline_run.ontology_id, annotator)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _print_runs(report: dict, baseline: int) -> None:
    for run_id, run in report["runs"].items():
        verified = run["calendar"]["verified"]
        cost = run["cost"]
        typer.echo(f"run {run_id}{' (baseline)' if run_id == baseline else ''}")
        typer.echo(
            f"  events    raw {rate_text(run['calendar']['event_recall'])}   "
            f"verified {rate_text(verified['event_recall'])}   "
            f"false alarms {rate_text(verified['false_alarm_rate'])}"
        )
        if "article" in run:
            scores = run["article"]["end_to_end"]
            typer.echo(
                f"  articles  P {scores['precision'] or 0:.3f} {scores['precision_ci']}  "
                f"R {scores['recall'] or 0:.3f} {scores['recall_ci']}  "
                f"F1 {scores['f1'] or 0:.3f} {scores['f1_ci']}"
            )
        typer.echo(
            f"  cost      {cost['pairs']} verdicts, "
            f"{cost['input_tokens'] + cost['output_tokens']} tokens, "
            f"{cost['seconds'] / 3600:.1f} h"
        )


def _print_sample(sample: dict | None) -> None:
    if not sample:
        return
    typer.echo(f"sample    {sample['labelled_prefix']} labelled in order")
    if sample["out_of_turn"]:
        typer.secho(
            f"          {len(sample['out_of_turn'])} labelled out of turn (not counted)",
            fg=typer.colors.YELLOW,
        )
    if "status" in sample:
        status = sample["status"]
        typer.echo(
            f"stopping  half-widths {status['half_widths']} vs ±{status['target']}: "
            f"{'target met' if status['target_met'] else 'keep labelling'}"
        )


def _print_comparisons(report: dict) -> None:
    for arm_id, comparison in report["comparisons"].items():
        f1 = comparison["f1"]
        typer.echo(
            f"arm {arm_id} vs baseline: F1 difference {f1['difference']} {f1['difference_ci']}"
            f" -> {'improvement' if f1['improvement'] else 'not shown'}"
        )
