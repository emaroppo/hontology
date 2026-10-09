"""Leaderboard: one row per run, scored end to end on the labelled sample."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.home.data import Context, clear_run_caches
from hontology.apps.ui.shared import ci_text


def show_leaderboard(ctx: Context) -> None:
    board = ctx.board
    labelled = board["labelled_articles"]
    if not labelled:
        st.info(
            "No article of this sample is labelled yet, so nothing can be scored. Label "
            "some on the **Labelling** page, or score against a machine annotation set "
            "(Truth, in Settings at the top right)."
        )
        return
    cols = st.columns([3, 2, 1])
    show_partial = cols[0].toggle(
        "Include runs that did not cover the sample",
        key="home_partial",
        help="A run made for another calendar processed few of these articles, so its "
        "recall here says nothing about it. Scoring them takes longer.",
    )
    cols[1].caption(f"Computed at {board['computed_at']}; kept for five minutes.")
    if cols[2].button("Recompute", help="After new labels or runs."):
        st.session_state["home_recompute"] = st.session_state.get("home_recompute", 0) + 1
        clear_run_caches()
        st.rerun()
    shown = [r for r in ctx.runs if show_partial or r["covered"] == labelled]
    shown.sort(key=lambda r: -((r["end_to_end"] or {}).get("f1") or -1))
    if not shown:
        st.info("No run covered every labelled article. Include partial runs to see them.")
        return
    st.dataframe([_row(r, labelled, ctx.calendar_cache) for r in shown], hide_index=True)
    st.caption(
        f"End to end on the first {board['labelled_prefix']} articles of the sample, "
        f"{labelled} of them labelled: a leaf a run never judged counts as no, and each "
        "article is weighted by its calendar window, as in the arms report. Intervals are "
        "95%, resampling articles. Tokens and seconds are judging on these articles."
        + (
            f" {board['out_of_turn']} article(s) labelled out of order are left out."
            if board["out_of_turn"]
            else ""
        )
    )
    _calendar_scoring(ctx, shown)


def _row(r: dict, labelled: int, calendar_cache: dict) -> dict:
    """One run's leaderboard row."""
    e2e = r["end_to_end"] or {}
    cost = r["cost"] or {}
    calendar = calendar_cache.get(r["run_id"])
    recall = (calendar or {}).get("event_recall") or {}
    return {
        "Run": f"{r['run_id']} · {r['name']}",
        "Prompt": r["versions"]["prompt_id"],
        "F1": ci_text(e2e.get("f1"), e2e.get("f1_ci")),
        "Precision": ci_text(e2e.get("precision"), e2e.get("precision_ci")),
        "Recall": ci_text(e2e.get("recall"), e2e.get("recall_ci")),
        "Covered": f"{r['covered']}/{labelled}",
        "Pairs judged": r["pairs_judged"],
        "Tokens": (cost.get("input_tokens") or 0) + (cost.get("output_tokens") or 0),
        "Seconds": round(cost.get("seconds") or 0),
        "Calendar recall": ci_text(recall.get("rate"), recall.get("ci")) if calendar else "",
        "Filter": ", ".join(r["versions"]["filter"]) or "not recorded",
        "Retrieval": r["versions"]["retrieval"],
        "Judge": r["versions"]["judge"],
    }


def _calendar_scoring(ctx: Context, shown: list[dict]) -> None:
    """Score chosen runs on their calendars, on request: tens of seconds each."""
    calendar_cache = ctx.calendar_cache
    with_calendar = [r for r in shown if r["calendar"] and r["run_id"] not in calendar_cache]
    if not with_calendar:
        return
    cols = st.columns([3, 1])
    pick = cols[0].multiselect(
        "Add calendar recall (runs that worked through a calendar; tens of seconds each)",
        [r["run_id"] for r in with_calendar],
        format_func=lambda i: f"run {i}",
    )
    if pick and cols[1].button("Score on their calendars"):
        for run_id in pick:
            try:
                with st.spinner(f"Run {run_id}: walking its calendar…"):
                    calendar_cache[run_id] = ctx.api.run_calendar(run_id)
            except ApiError as exc:
                st.error(exc.detail)
        st.rerun()
