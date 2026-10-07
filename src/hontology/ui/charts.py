"""Reusable figures for the evaluation pages.

Altair rather than a new plotting dependency — Streamlit already ships it.

Two conventions, both about not overstating what the data supports:

- **Intervals are drawn wherever they exist.** A bar with an error bar reads as
  an estimate; a bare bar reads as a fact. Most of these numbers are estimates
  from a few hundred labels.
- **Counts are shown alongside rates.** A slice with four labelled pairs gets the
  same visual weight as one with four hundred unless the chart says otherwise,
  so `n` is on the axis label or the tooltip everywhere.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

# A single accent plus muted greys, so charts read as one system rather than a
# palette per page.
ACCENT = "#2e7d5b"
MUTED = "#9aa0a6"
WARN = "#b3541e"


def _empty(message: str) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"message": [message]}))
        .mark_text(size=13, color=MUTED)
        .encode(text="message:N")
        .properties(height=90)
    )


def metric_with_interval(rows: list[dict], *, metric: str = "f1", title: str = "") -> alt.Chart:
    """Per-slice metric as bars with their confidence intervals.

    Used for breakdowns, where the whole point is that some slices are measured
    far more thinly than others.
    """
    usable = [r for r in rows if r.get(metric) is not None]
    if not usable:
        return _empty(f"no slice has a measurable {metric}")

    ci_key = f"{metric}_ci"
    frame = pd.DataFrame(
        [
            {
                "label": r["label"],
                "value": r[metric],
                "n": r.get("n", 0),
                "low": (r.get(ci_key) or (None, None))[0],
                "high": (r.get(ci_key) or (None, None))[1],
            }
            for r in usable
        ]
    )

    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=["label:N", "value:Q", "n:Q", "low:Q", "high:Q"],
    )
    bars = base.mark_bar(color=ACCENT, size=14).encode(
        x=alt.X("value:Q", title=metric, scale=alt.Scale(domain=[0, 1]))
    )
    if frame["low"].notna().any():
        error = base.mark_rule(color=MUTED, strokeWidth=2).encode(x="low:Q", x2="high:Q")
        chart = bars + error
    else:
        chart = bars
    return chart.properties(
        title=title or f"{metric} by slice", height=max(120, 24 * len(frame))
    )


def recall_at_k(recall: dict[Any, float | None], *, title: str = "recall@k") -> alt.Chart:
    """Ranking quality before the cutoff."""
    usable = {int(k): v for k, v in recall.items() if v is not None}
    if not usable:
        return _empty("recall@k is undefined — the label set has no positives")

    frame = pd.DataFrame({"k": list(usable), "recall": list(usable.values())})
    return (
        alt.Chart(frame)
        .mark_line(point=True, color=ACCENT)
        .encode(
            x=alt.X("k:O", title="k"),
            y=alt.Y("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1])),
            tooltip=["k:O", "recall:Q"],
        )
        .properties(title=title, height=220)
    )


def calibration(bins: list[dict], *, title: str = "calibration") -> alt.Chart:
    """Observed accuracy per confidence bin, against the ideal diagonal.

    The diagonal is the point of the chart: bars that sit far from it mean the
    model's confidence is not a probability, however reasonable it looks.
    """
    usable = [b for b in bins if b.get("n")]
    if not usable:
        return _empty("no confidence values recorded for this run")

    frame = pd.DataFrame(
        [
            {
                "midpoint": (b["range"][0] + b["range"][1]) / 2,
                "accuracy": b["accuracy"],
                "n": b["n"],
            }
            for b in usable
            if b["accuracy"] is not None
        ]
    )
    if frame.empty:
        return _empty("no confidence bin has a measurable accuracy")

    bars = (
        alt.Chart(frame)
        .mark_bar(color=ACCENT, size=28)
        .encode(
            x=alt.X("midpoint:Q", title="stated confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy:Q", title="observed accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["midpoint:Q", "accuracy:Q", "n:Q"],
        )
    )
    ideal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[5, 5], color=MUTED)
        .encode(x="x:Q", y="y:Q")
    )
    return (bars + ideal).properties(title=title, height=240)


def confusion_bar(judge: dict, *, title: str = "outcomes") -> alt.Chart:
    """The four confusion counts, which are what the rates are built from."""
    frame = pd.DataFrame(
        [
            {"outcome": "true positive", "count": judge.get("tp", 0)},
            {"outcome": "true negative", "count": judge.get("tn", 0)},
            {"outcome": "false positive", "count": judge.get("fp", 0)},
            {"outcome": "false negative", "count": judge.get("fn", 0)},
        ]
    )
    if frame["count"].sum() == 0:
        return _empty("no labelled pairs were judged by this run")

    return (
        alt.Chart(frame)
        .mark_bar(size=26)
        .encode(
            y=alt.Y("outcome:N", sort=None, title=None),
            x=alt.X("count:Q", title="pairs"),
            # Errors in the warning colour, correct answers in the accent.
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=[
                        "true positive",
                        "true negative",
                        "false positive",
                        "false negative",
                    ],
                    range=[ACCENT, ACCENT, WARN, WARN],
                ),
                legend=None,
            ),
            tooltip=["outcome:N", "count:Q"],
        )
        .properties(title=title, height=150)
    )


def leaderboard_scatter(rows: list[dict], *, title: str = "F1 against cost") -> alt.Chart:
    """Quality against tokens spent, so a marginal gain's price is visible."""
    usable = [r for r in rows if r.get("f1") is not None]
    if not usable:
        return _empty("no recorded run has a measurable F1")

    frame = pd.DataFrame(
        [
            {
                "run": f"{r['run_id']}: {r.get('run_name', '')}",
                "f1": r["f1"],
                "tokens": (r.get("output_tokens") or 0) + (r.get("input_tokens") or 0),
                "n": r.get("n_labels", 0),
                "prompt": r.get("prompt_id") or "",
            }
            for r in usable
        ]
    )
    return (
        alt.Chart(frame)
        .mark_circle(size=140, color=ACCENT, opacity=0.75)
        .encode(
            x=alt.X("tokens:Q", title="tokens used"),
            y=alt.Y("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
            tooltip=["run:N", "f1:Q", "tokens:Q", "n:Q", "prompt:N"],
        )
        .properties(title=title, height=260)
    )
