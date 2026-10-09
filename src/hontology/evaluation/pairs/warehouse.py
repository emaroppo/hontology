"""The leaderboard: run results kept somewhere they can be compared over time.

Metrics live in DuckDB rather than in the operational database because the
questions asked of them are analytical — rank these twenty runs, show F1 against
token cost, group by prompt — and because a leaderboard is a derived artifact.
Losing it costs a recompute, not data.

**Counts are stored, not just rates.** A rate cannot be re-aggregated: averaging
the F1 of two runs is not the F1 of their union. Keeping tp/fp/tn/fn means any
grouping can recompute its own rates correctly.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from hontology.config import get_settings
from hontology.evaluation.metrics.intervals import format_ci, format_num
from hontology.evaluation.pairs.evaluate import RunEvaluation

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id             INTEGER,
    run_name           VARCHAR,
    ontology_version   VARCHAR,
    recorded_at        TIMESTAMP DEFAULT current_timestamp,
    n_labels           INTEGER,
    tp INTEGER, fp INTEGER, tn INTEGER, fn INTEGER,
    precision          DOUBLE,
    recall             DOUBLE,
    f1                 DOUBLE,
    f1_lo              DOUBLE,
    f1_hi              DOUBLE,
    retrieval_coverage DOUBLE,
    retrieval_mrr      DOUBLE,
    cutoff_recall      DOUBLE,
    liveness_clean     INTEGER,
    liveness_errors    INTEGER,
    candidates_key     VARCHAR,
    judge_key          VARCHAR,
    provider           VARCHAR,
    model              VARCHAR,
    prompt_id          VARCHAR,
    input_tokens       BIGINT,
    output_tokens      BIGINT
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    settings = get_settings()
    target = path or settings.warehouse_path
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(target))
    connection.execute(SCHEMA)
    return connection


def record(
    evaluation: RunEvaluation,
    *,
    config: dict,
    candidates_key: str,
    judge_key: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    path: Path | None = None,
) -> None:
    """Append one run's results. Re-recording a run replaces its row."""
    judge = evaluation.judge
    retrieval = evaluation.retrieval
    f1_lo, f1_hi = judge.get("f1_ci", (None, None))
    judge_config = config.get("judge", {})

    connection = connect(path)
    try:
        connection.execute("DELETE FROM run_metrics WHERE run_id = ?", [evaluation.run_id])
        connection.execute(
            """
            INSERT INTO run_metrics (
                run_id, run_name, ontology_version, n_labels,
                tp, fp, tn, fn, precision, recall, f1, f1_lo, f1_hi,
                retrieval_coverage, retrieval_mrr, cutoff_recall,
                liveness_clean, liveness_errors,
                candidates_key, judge_key, provider, model, prompt_id,
                input_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                evaluation.run_id,
                evaluation.run_name,
                evaluation.ontology_version,
                evaluation.n_labels,
                judge["tp"],
                judge["fp"],
                judge["tn"],
                judge["fn"],
                judge["precision"],
                judge["recall"],
                judge["f1"],
                f1_lo,
                f1_hi,
                retrieval["coverage"],
                retrieval["mrr"],
                retrieval["cutoff_recall"],
                evaluation.liveness["clean"],
                evaluation.liveness["errors"],
                candidates_key,
                judge_key,
                judge_config.get("provider"),
                judge_config.get("model"),
                judge_config.get("prompt_id"),
                input_tokens,
                output_tokens,
            ],
        )
    finally:
        connection.close()


def leaderboard(path: Path | None = None, *, limit: int = 50) -> list[dict]:
    """Runs ranked by F1, with the interval and the denominator alongside.

    Both are shown deliberately: an F1 without its label count and its interval
    invites ranking runs by differences that are not there.
    """
    connection = connect(path)
    try:
        rows = connection.execute(
            """
            SELECT run_id, run_name, ontology_version, n_labels,
                   tp, fp, tn, fn, precision, recall, f1, f1_lo, f1_hi,
                   retrieval_coverage, cutoff_recall,
                   liveness_errors, provider, model, prompt_id,
                   input_tokens, output_tokens, recorded_at
            FROM run_metrics
            ORDER BY f1 DESC NULLS LAST, n_labels DESC
            LIMIT ?
            """,
            [limit],
        )
        columns = [d[0] for d in rows.description]
        return [dict(zip(columns, row, strict=True)) for row in rows.fetchall()]
    finally:
        connection.close()


def format_leaderboard(rows: list[dict]) -> str:
    if not rows:
        return "(no runs recorded yet)"

    lines = [
        f"{'run':>4}  {'name':<24} {'n':>5}  {'F1':>6}  {'interval':<16} "
        f"{'prec':>6} {'rec':>6}  {'prompt':<18} {'errs':>5}"
    ]
    lines.append("-" * 104)
    for row in rows:
        interval = format_ci(row["f1_lo"], row["f1_hi"])
        lines.append(
            f"{row['run_id']:>4}  {(row['run_name'] or '')[:24]:<24} "
            f"{row['n_labels']:>5}  {format_num(row['f1'], 6)}  {interval:<16} "
            f"{format_num(row['precision'], 6)} {format_num(row['recall'], 6)}  "
            f"{(row['prompt_id'] or '')[:18]:<18} {row['liveness_errors']:>5}"
        )
    return "\n".join(lines)
