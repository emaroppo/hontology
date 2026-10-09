"""`hontology eval`: score runs against the ground-truth bank."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from hontology.cli.common import (
    After,
    Before,
    IncludeMachine,
    emit,
    load_run,
    log_to_console,
    read_json,
    write_csv,
)
from hontology.db.session import session_scope

app = typer.Typer(help="Score runs against the ground-truth bank.")

# The calendar's headline rates, in the order reports show them.
CALENDAR_RATES = (
    ("event recall", "event_recall"),
    ("precursor recall", "precursor_recall"),
    ("false alarm rate", "false_alarm_rate"),
)
REVIEW_COLUMNS = (
    "entry_id",
    "kind",
    "concept",
    "countries",
    "date",
    "description",
    "first_seen",
    "url",
    "evidence",
    "confirmed",
    "note",
)


def _rate(stat: dict, sep: str = " ") -> str:
    value = "-" if stat["rate"] is None else f"{stat['rate']:.2f}"
    return f"{stat['hits']}/{stat['n']}{sep}{value}"


@app.command("run")
def eval_run(
    run_id: int,
    include_machine: bool = typer.Option(
        False, help="Count un-adjudicated machine labels. Off by default."
    ),
    include_stale: bool = typer.Option(
        False, help="Count labels whose concept was reworded after labelling."
    ),
    record: bool = typer.Option(False, help="Also write the result to the leaderboard."),
) -> None:
    """Per-stage metrics for one run, with intervals and denominators."""
    from hontology.evalkit import evaluate, warehouse

    with session_scope() as session:
        evaluation = evaluate.evaluate_run(
            session, run_id, include_machine=include_machine, include_stale=include_stale
        )
        typer.echo(evaluate.format_report(evaluation))
        if record:
            run = load_run(session, run_id)
            warehouse.record(
                evaluation,
                config=run.config,
                candidates_key=run.candidates_key,
                judge_key=run.judge_key,
            )
            typer.secho("recorded to the leaderboard", fg=typer.colors.GREEN)


@app.command("breakdown")
def eval_breakdown(
    run_id: int,
    dimension: str = typer.Option("concept", help="concept | family | category | locus"),
    include_machine: IncludeMachine = False,
) -> None:
    """Metrics sliced, so a systematic failure is visible rather than pooled away."""
    from hontology.evalkit import breakdown

    with session_scope() as session:
        rows = breakdown.breakdown(
            session, run_id, dimension=dimension, include_machine=include_machine
        )
    typer.echo(breakdown.format_breakdown(rows, dimension))


@app.command("errors")
def eval_errors(
    run_id: int,
    kind: str | None = typer.Option(None, help="false_positive | false_negative"),
    include_machine: IncludeMachine = False,
    limit: int = typer.Option(20, help="How many to show."),
) -> None:
    """Every misclassified pair, with the model's evidence and reasoning."""
    from hontology.evalkit import errors

    with session_scope() as session:
        rows = errors.triage(
            session, run_id, kind=kind, include_machine=include_machine, limit=limit
        )
        stats = errors.summary(session, run_id, include_machine=include_machine)
    typer.echo(errors.format_triage(rows))
    typer.echo(
        f"{stats['total']} error(s): {stats['false_positives']} FP, "
        f"{stats['false_negatives']} FN"
    )
    for concept, counts in list(stats["by_concept"].items())[:5]:
        typer.echo(
            f"  {concept:<30} FP={counts['false_positive']} FN={counts['false_negative']}"
        )


@app.command("filter-report")
def eval_filter_report(ontology_id: int) -> None:
    """Per-code cost and benefit of the pre-scrape filter mapping."""
    from hontology.evalkit import filter_report

    with session_scope() as session:
        typer.echo(filter_report.format_report(filter_report.report(session, ontology_id)))


@app.command("sweep")
def eval_sweep(
    ontology_id: int,
    sweep_path: Path,
    execute: bool = typer.Option(False, help="Run the plan, not just print it."),
    documents: int = typer.Option(50, help="Documents per cell."),
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged per cell."),
) -> None:
    """Plan or run a parameter sweep. Cells already run are skipped."""
    from hontology.evalkit import sweep as sweep_module

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


@app.command("funnel")
def eval_funnel(run_id: int) -> None:
    """Where the volume went, stage by stage. Needs no labels."""
    from hontology.evalkit import funnel

    with session_scope() as session:
        typer.echo(funnel.format_funnel(funnel.funnel(session, run_id)))


@app.command("calendar")
def eval_calendar(
    run_id: int,
    calendar_path: Path,
    before: Before = 1,
    after: After = 2,
    out: Path | None = typer.Option(None, help="Also write the full result as JSON."),
) -> None:
    """Event-level results against a calendar of known events. Needs no labels."""
    from hontology.evalkit import calendar, calendar_score
    from hontology.evalkit.metrics import format_ci

    with session_scope() as session:
        result = calendar_score.evaluate(
            session, run_id, calendar.load(calendar_path), before=before, after=after
        )
    if out is not None:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

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

    typer.echo("")
    verified = result["summary"]["verified"]

    def rates(stats: dict, indent: str) -> None:
        for label, key in CALENDAR_RATES:
            stat = stats[key]
            typer.echo(f"{indent + label:18} {_rate(stat, '  ')}  {format_ci(*stat['ci'])}")

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


@app.command("arms")
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
    from hontology.db.models import Run
    from hontology.evalkit import annotations, arms, arms_report, calendar
    from hontology.evalkit.document_labels import document_label_map

    if labels_path is not None and annotator is not None:
        raise typer.BadParameter("give --labels or --annotator, not both")
    with session_scope() as session:
        labels = None
        if labels_path is not None or annotator is not None:
            baseline_run = session.get(Run, baseline)
            if baseline_run is None:
                raise typer.BadParameter(f"run {baseline} does not exist")
            if labels_path is not None:
                labels = document_label_map(
                    session, baseline_run.ontology_id, labels_path.read_text(encoding="utf-8")
                )
            else:
                try:
                    labels = annotations.truth(session, baseline_run.ontology_id, annotator)
                except LookupError as exc:
                    raise typer.BadParameter(str(exc)) from exc
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

    for run_id, run in report["runs"].items():
        verified = run["calendar"]["verified"]
        cost = run["cost"]
        typer.echo(f"run {run_id}{' (baseline)' if run_id == baseline else ''}")
        typer.echo(
            f"  events    raw {_rate(run['calendar']['event_recall'])}   "
            f"verified {_rate(verified['event_recall'])}   "
            f"false alarms {_rate(verified['false_alarm_rate'])}"
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
    sample = report.get("sample")
    if sample:
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
    for arm_id, comparison in report["comparisons"].items():
        f1 = comparison["f1"]
        typer.echo(
            f"arm {arm_id} vs baseline: F1 difference {f1['difference']} {f1['difference_ci']}"
            f" -> {'improvement' if f1['improvement'] else 'not shown'}"
        )


@app.command("calendar-review-export")
def eval_calendar_review_export(
    run_id: int,
    calendar_path: Path,
    out: Path = typer.Option(..., help="CSV to write; fill `confirmed` with yes or no."),
    per_entry: int = typer.Option(5, help="Unreviewed matches listed per entry."),
    before: Before = 1,
    after: After = 2,
) -> None:
    """Write the matches still awaiting review, earliest first, for a person to mark.

    For an event or precursor, `confirmed` asks: does this article describe this
    very event? For a control: does it report a real instance of the concept
    there and then? Entries already confirmed are left out.
    """
    from hontology.evalkit import calendar, calendar_score

    entries = calendar.load(calendar_path)
    by_id = {e.id: e for e in entries}
    with session_scope() as session:
        result = calendar_score.evaluate(session, run_id, entries, before=before, after=after)
    rows = [
        {
            **{key: row[key] for key in ("kind", "concept", "countries", "date")},
            "entry_id": row["id"],
            "description": by_id[row["id"]].description,
            "first_seen": match["first_seen"],
            "url": match["url"],
            "evidence": (match["evidence"] or "").replace("\n", " "),
            "confirmed": "",
            "note": "",
        }
        for row in result["entries"]
        if row["verified"] is None
        for match in [m for m in row["matches"] if m["confirmed"] is None][:per_entry]
    ]
    write_csv(out, REVIEW_COLUMNS, rows)
    typer.echo(
        f"{len(rows)} match(es) to review across {len({r['entry_id'] for r in rows})} entries"
    )


@app.command("calendar-review-import")
def eval_calendar_review_import(path: Path) -> None:
    """Read reviewed matches back; rows left blank in `confirmed` are skipped."""
    import csv

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from hontology.db.models import CalendarReview

    answers = dict.fromkeys(("yes", "y", "true", "1"), True)
    answers |= dict.fromkeys(("no", "n", "false", "0"), False)
    saved = skipped = 0
    with path.open(encoding="utf-8", newline="") as handle, session_scope() as session:
        for row in csv.DictReader(handle):
            answer = answers.get((row.get("confirmed") or "").strip().lower())
            if answer is None:
                skipped += 1
                continue
            statement = pg_insert(CalendarReview).values(
                entry_id=row["entry_id"].strip(),
                document_url=row["url"].strip(),
                confirmed=answer,
                note=(row.get("note") or "").strip() or None,
            )
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=["entry_id", "document_url"],
                    set_={"confirmed": answer, "note": statement.excluded.note},
                )
            )
            saved += 1
    typer.echo(f"saved {saved} review(s); {skipped} row(s) left blank")


@app.command("detections")
def eval_detections(
    run_id: int,
    out: Path | None = typer.Option(None, help="Write CSV to a file instead of stdout."),
    events: bool = typer.Option(False, help="Aggregate to one row per (concept, locus, date)."),
    min_confidence: float | None = typer.Option(None, help="Drop weaker detections."),
    verified_only: bool = typer.Option(False, help="Only detections a human has confirmed."),
) -> None:
    """Export what the pipeline found. Every row carries its verification status."""
    from hontology.evalkit import detections as detections_module

    exporter = (
        detections_module.export_events if events else detections_module.export_detections
    )
    with session_scope() as session:
        text_out = exporter(
            session, run_id, min_confidence=min_confidence, verified_only=verified_only
        )
        stats = detections_module.summary(session, run_id)
    emit(text_out, out, count_rows=True)

    typer.echo(
        f"{stats['detections']} detection(s) over {stats['events']} event(s); "
        f"{stats['by_verification']}"
    )
    if stats["by_verification"].get("unverified"):
        typer.secho(
            "unverified detections are model claims, not facts — adjudicate before "
            "treating them as findings",
            fg=typer.colors.YELLOW,
        )


@app.command("compare")
def eval_compare(run_a: int, run_b: int, include_machine: IncludeMachine = False) -> None:
    """Paired comparison of two runs over the pairs they both judged."""
    from hontology.evalkit import compare

    with session_scope() as session:
        result = compare.compare_runs(session, run_a, run_b, include_machine=include_machine)

    typer.echo(f"shared labelled pairs  {result.n_shared_labelled}")
    typer.echo(
        f"discordant             {result.paired['discordant']} "
        f"(only A right: {result.paired['only_a_correct']}, "
        f"only B right: {result.paired['only_b_correct']})"
    )
    p = result.paired["p_value"]
    typer.echo(f"p-value                {'—' if p is None else f'{p:.4f}'}")
    typer.secho(f"\n{result.verdict}", bold=True)


@app.command("consistency")
def eval_consistency(run_id: int, against: int | None = None) -> None:
    """Determinism against another run, and agreement across documents."""
    from hontology.evalkit import compare

    with session_scope() as session:
        if against is not None:
            result = compare.determinism(session, run_id, against)
            status = "reproducible" if result["reproducible"] else "NOT reproducible"
            typer.secho(
                f"determinism: {status} "
                f"({result['n_shared']} shared, {result['flipped']} flipped)",
                fg=typer.colors.GREEN if result["reproducible"] else typer.colors.RED,
            )
            if not result["reproducible"]:
                typer.echo("  run-to-run noise is a floor under every A/B delta; a smaller")
                typer.echo("  difference than this cannot be attributed to a config change")
        agreement = compare.cross_document_agreement(session, run_id)
    typer.echo(
        f"cross-document: {agreement['multi_document_groups']} group(s), "
        f"{agreement['split_groups']} split, mean {agreement['mean_agreement']}"
    )


@app.command("baseline")
def eval_baseline(
    run_id: int, out: Path = Path("baseline.json"), tolerance: float = 0.05
) -> None:
    """Record a run's metrics as the floor future runs must clear."""
    from hontology.evalkit import regression

    with session_scope() as session:
        baseline = regression.capture_baseline(session, run_id, tolerance=tolerance)
    regression.save_baseline(baseline, out)
    typer.echo(f"baseline from run {run_id} -> {out}")


@app.command("gate")
def eval_gate(run_id: int, baseline_path: Path = Path("baseline.json")) -> None:
    """Regression gate: liveness first, then metric floors. Exits non-zero on failure."""
    from hontology.evalkit import regression

    baseline = regression.load_baseline(baseline_path)
    with session_scope() as session:
        result = regression.check(session, run_id, baseline)

    live = result["liveness"]
    typer.echo(f"liveness  {live['clean']}/{live['verdicts']} clean, {live['errors']} error(s)")
    for floor in result["floors"]:
        actual = floor["actual"]
        shown = f"{actual:.3f}" if actual is not None else "—"
        mark = "ok " if floor["ok"] else "FAIL"
        limit = f"{floor['limit']:.3f}" if floor.get("limit") is not None else "—"
        typer.echo(f"{mark}      {floor['metric']:<10} {shown}  (floor {limit})")

    if result["ok"]:
        typer.secho(
            f"\ngate passed ({result['n_judged_labelled']} labelled pairs checked)",
            fg=typer.colors.GREEN,
        )
        return
    inconclusive = result["inconclusive"]
    typer.secho(
        "\ngate INCONCLUSIVE — nothing to check" if inconclusive else "\ngate FAILED",
        fg=typer.colors.YELLOW if inconclusive else typer.colors.RED,
    )
    for failure in result["failures"]:
        typer.echo(f"  - {failure}")
    raise typer.Exit(2 if inconclusive else 1)


@app.command("leaderboard")
def eval_leaderboard(limit: int = 50) -> None:
    """Recorded runs, ranked by F1, with intervals and denominators."""
    from hontology.evalkit import warehouse

    typer.echo(warehouse.format_leaderboard(warehouse.leaderboard(limit=limit)))
