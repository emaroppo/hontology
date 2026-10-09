"""`hontology eval` regression tooling: baselines, the gate, the leaderboard."""

from __future__ import annotations

from pathlib import Path

import typer

from hontology.db.session import session_scope


def eval_baseline(
    run_id: int, out: Path = Path("baseline.json"), tolerance: float = 0.05
) -> None:
    """Record a run's metrics as the floor future runs must clear."""
    from hontology.evaluation.pairs import regression

    with session_scope() as session:
        baseline = regression.capture_baseline(session, run_id, tolerance=tolerance)
    regression.save_baseline(baseline, out)
    typer.echo(f"baseline from run {run_id} -> {out}")


def eval_gate(run_id: int, baseline_path: Path = Path("baseline.json")) -> None:
    """Regression gate: liveness first, then metric floors. Exits non-zero on failure."""
    from hontology.evaluation.pairs import regression

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


def eval_leaderboard(limit: int = 50) -> None:
    """Recorded runs, ranked by F1, with intervals and denominators."""
    from hontology.evaluation.pairs import warehouse

    typer.echo(warehouse.format_leaderboard(warehouse.leaderboard(limit=limit)))
