"""Single run: one run end to end, with a couple of numbers per stage; each
stage's own evaluation lives on its page."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.home.data import Context, run_detections, run_funnel
from hontology.apps.ui.home.stages import by_stage
from hontology.apps.ui.shared import ci_text


def show_single(ctx: Context) -> None:
    labels = {r["run_id"]: f"{r['run_id']} · {r['name']} ({r['status']})" for r in ctx.runs}
    run_id = ctx.params.run_id
    if run_id is None or run_id not in labels:
        st.info("Pick a run that has judged something in Settings, at the top right.")
        return
    st.markdown(f"**{labels[run_id]}**")
    row = next(r for r in ctx.runs if r["run_id"] == run_id)
    labelled = ctx.board["labelled_articles"]
    if row["end_to_end"] is None and row["covered"] < labelled:
        st.info(
            f"Run {run_id} worked on {row['covered']} of the {labelled} "
            "labelled articles, so it is not scored on this sample. Turn on "
            '"Include runs that did not cover the sample" on the Leaderboard tab to '
            "score it anyway."
        )

    st.markdown("**End to end**")
    e2e = row["end_to_end"] or {}
    cols = st.columns(4)
    cols[0].metric("F1", ci_text(e2e.get("f1"), e2e.get("f1_ci")))
    cols[1].metric("Precision", ci_text(e2e.get("precision"), e2e.get("precision_ci")))
    cols[2].metric("Recall", ci_text(e2e.get("recall"), e2e.get("recall_ci")))
    cols[3].metric("Sample covered", f"{row['covered']}/{labelled}")
    if row["calendar"]:
        _calendar(ctx, run_id, row["calendar"])

    sample = by_stage(ctx, run_id, row)
    _disagreements(sample)
    _funnel(run_id)
    _detections(run_id)


def _calendar(ctx: Context, run_id: int, calendar_name: str) -> None:
    """The run's calendar scores, or a button to compute them."""
    calendar = ctx.calendar_cache.get(run_id)
    if calendar is None:
        if st.button(f"Score on its calendar ({calendar_name}; tens of seconds)"):
            try:
                with st.spinner("Walking the calendar…"):
                    ctx.calendar_cache[run_id] = ctx.api.run_calendar(run_id)
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
        return
    cols = st.columns(4)
    for col, name, key in (
        (cols[0], "Event recall", "event_recall"),
        (cols[1], "Precursor recall", "precursor_recall"),
        (cols[2], "False alarms on controls", "false_alarm_rate"),
    ):
        stat = calendar.get(key) or {}
        col.metric(name, ci_text(stat.get("rate"), stat.get("ci")))
    verified = (calendar.get("verified") or {}).get("event_recall") or {}
    cols[3].metric(
        "Verified event recall",
        ci_text(verified.get("rate"), verified.get("ci")),
    )
    if calendar.get("positives_lost_at"):
        st.caption(
            "Events missed, by the stage that lost them: "
            + ", ".join(f"{k} {v}" for k, v in calendar["positives_lost_at"].items())
        )


def _disagreements(sample: dict) -> None:
    with st.expander("Where it disagrees with the labels"):
        try:
            errors = sample.get("errors") or []
            st.dataframe(
                [
                    {
                        "Position": e["position"],
                        "Kind": e["kind"],
                        "Class": e["concept"],
                        "Article": e["document_title"] or e["document_url"],
                        "Link": e["document_url"],
                    }
                    for e in errors
                ],
                hide_index=True,
                column_config={"Link": st.column_config.LinkColumn(display_text="open")},
            )
        except ApiError as exc:
            st.error(exc.detail)


def _funnel(run_id: int) -> None:
    with st.expander("Funnel: where the volume went"):
        try:
            flow = run_funnel(run_id)
            fcols = st.columns(2)
            for col, key in ((fcols[0], "documents"), (fcols[1], "pairs")):
                for step in flow[key]:
                    share = (
                        f"{step['of_previous']:.1%} of previous"
                        if step["of_previous"] is not None
                        else "start"
                    )
                    col.markdown(f"`{step['count']:>9,}`  {step['name']}: {share}")
        except ApiError as exc:
            st.error(exc.detail)


def _detections(run_id: int) -> None:
    with st.expander("Detections"):
        try:
            found, found_csv = run_detections(run_id)
            stats = found["summary"]
            dcols = st.columns(3)
            dcols[0].metric("Detections", stats["detections"])
            dcols[1].metric("Events", stats["events"], help="Distinct (class, place, date).")
            dcols[2].metric("Confirmed", stats["by_verification"].get("confirmed", 0))
            if found["rows"]:
                st.download_button(
                    "Download detections (CSV)",
                    data=found_csv,
                    file_name=f"detections-run{run_id}.csv",
                    mime="text/csv",
                )
        except ApiError as exc:
            st.error(exc.detail)
