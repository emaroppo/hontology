"""The regression gate.

Two checks, in this order, and the order is the point.

**1. Liveness, first and independently.** Did every pair produce a usable
verdict? A malformed-output bug does not move precision or recall — it removes
pairs from the denominator entirely, so accuracy metrics stay perfectly healthy
while the run silently measures a smaller and smaller subset. Checking liveness
before looking at any rate is what catches that class of failure at all.

**2. Metric floors.** Precision, recall and F1 must not fall below a recorded
baseline. Floors, not equality: a small change in a local model's output should
not fail a build, but a collapse should.

A baseline is captured explicitly from a run you have decided is good, so the
gate compares against a deliberate choice rather than whatever happened last.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from hontology.evaluation.metrics.intervals import format_num
from hontology.evaluation.pairs import evaluate as evaluate_module


@dataclass
class Baseline:
    run_id: int
    run_name: str
    ontology_version: str
    n_labels: int
    min_precision: float | None
    min_recall: float | None
    min_f1: float | None
    # How much a metric may drop below the recorded value before failing. Local
    # models are not bit-reproducible across restarts, so a hard equality gate
    # would fail on noise and get switched off — which is worse than a loose one.
    tolerance: float = 0.05

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> Baseline:
        return Baseline(**json.loads(text))


def capture_baseline(session: Session, run_id: int, *, tolerance: float = 0.05) -> Baseline:
    """Record a run's metrics as the floor future runs must clear."""
    evaluation = evaluate_module.evaluate_run(session, run_id)
    judge = evaluation.judge
    return Baseline(
        run_id=run_id,
        run_name=evaluation.run_name,
        ontology_version=evaluation.ontology_version,
        n_labels=evaluation.n_labels,
        min_precision=judge["precision"],
        min_recall=judge["recall"],
        min_f1=judge["f1"],
        tolerance=tolerance,
    )


def check(session: Session, run_id: int, baseline: Baseline) -> dict:
    """Run both gates. Returns a result with ``ok`` and a per-check breakdown."""
    evaluation = evaluate_module.evaluate_run(session, run_id)
    failures: list[str] = []

    # --- gate 1: liveness -------------------------------------------------
    live = evaluation.liveness
    liveness_ok = bool(live["ok"])
    if not liveness_ok:
        failures.append(
            f"liveness: {live['errors']} error(s) and {live['unparsed']} unparsed "
            f"verdict(s) — the denominator is smaller than it looks"
        )

    # --- gate 2: metric floors -------------------------------------------
    judge = evaluation.judge
    floors: list[dict] = []
    for name, floor in (
        ("precision", baseline.min_precision),
        ("recall", baseline.min_recall),
        ("f1", baseline.min_f1),
    ):
        actual = judge[name]
        if floor is None:
            floors.append({"metric": name, "floor": None, "actual": actual, "ok": True})
            continue
        limit = floor - baseline.tolerance
        ok = actual is not None and actual >= limit
        floors.append(
            {"metric": name, "floor": floor, "limit": limit, "actual": actual, "ok": ok}
        )
        if not ok:
            shown = format_num(actual)
            failures.append(f"{name}: {shown} below floor {limit:.3f}")

    # --- gate 3: is there anything to check at all -------------------------
    #
    # A gate that passes because the bank is empty is worse than no gate: it
    # reports green while checking nothing. An empty denominator is an
    # inconclusive result, not a pass.
    judged = judge["n"]
    inconclusive = judged == 0
    if inconclusive:
        failures.append(
            "inconclusive: no trusted labels covered this run, so no metric could "
            "be checked. Label some pairs, or adjudicate the machine proposals."
        )

    return {
        "run_id": run_id,
        "ok": liveness_ok and all(f["ok"] for f in floors) and not inconclusive,
        "inconclusive": inconclusive,
        "liveness": live | {"ok": liveness_ok},
        "floors": floors,
        "n_labels": evaluation.n_labels,
        "n_judged_labelled": judged,
        "baseline_run": baseline.run_id,
        "failures": failures,
    }


def save_baseline(baseline: Baseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.to_json(), encoding="utf-8")


def load_baseline(path: Path) -> Baseline:
    return Baseline.from_json(path.read_text(encoding="utf-8"))
