"""`hontology eval` on judged pairs: per-run metrics, slices, errors, comparisons."""

from __future__ import annotations

import typer

from hontology.apps.cli.common import IncludeMachine, load_run
from hontology.db.session import session_scope


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
    from hontology.evaluation.pairs import evaluate, warehouse

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


def eval_breakdown(
    run_id: int,
    dimension: str = typer.Option("concept", help="concept | family | category | locus"),
    include_machine: IncludeMachine = False,
) -> None:
    """Metrics sliced, so a systematic failure is visible rather than pooled away."""
    from hontology.evaluation.pairs import breakdown

    with session_scope() as session:
        rows = breakdown.breakdown(
            session, run_id, dimension=dimension, include_machine=include_machine
        )
    typer.echo(breakdown.format_breakdown(rows, dimension))


def eval_errors(
    run_id: int,
    kind: str | None = typer.Option(None, help="false_positive | false_negative"),
    include_machine: IncludeMachine = False,
    limit: int = typer.Option(20, help="How many to show."),
) -> None:
    """Every misclassified pair, with the model's evidence and reasoning."""
    from hontology.evaluation.pairs import errors

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


def eval_compare(run_a: int, run_b: int, include_machine: IncludeMachine = False) -> None:
    """Paired comparison of two runs over the pairs they both judged."""
    from hontology.evaluation.pairs import compare

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


def eval_consistency(run_id: int, against: int | None = None) -> None:
    """Determinism against another run, and agreement across documents."""
    from hontology.evaluation.pairs import compare

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
