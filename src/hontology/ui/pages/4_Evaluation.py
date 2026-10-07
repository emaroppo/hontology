"""Evaluation and leaderboard.

Every rate on this page is shown with its denominator and its interval, because
the failure this project exists to prevent is reading a point estimate from a
small label bank as a finding.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import charts
from hontology.ui.client import Api, ApiError
from hontology.ui.glossary import describe

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")

api = Api()

st.title("📊 Evaluation")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

runs = api.list_runs()
if not runs:
    st.info("No runs yet. Execute one with `hontology run start`.")
    st.stop()

run_labels = {f"{r['id']}: {r['name']} ({r['status']})": r for r in runs}

with st.sidebar:
    selected = run_labels[st.selectbox("Run", list(run_labels))]
    include_machine = st.toggle(
        "Count machine labels",
        value=False,
        help="Un-adjudicated machine proposals. Off by default: scoring an LLM "
        "against another LLM's unreviewed labels measures agreement, not correctness.",
    )
    include_stale = st.toggle(
        "Count stale labels",
        value=False,
        help="Labels whose concept was reworded after they were made.",
    )

try:
    evaluation = api.evaluate_run(
        selected["id"], include_machine=include_machine, include_stale=include_stale
    )
except ApiError as exc:
    st.error(exc.detail)
    st.stop()

judge = evaluation["judge"]
retrieval = evaluation["retrieval"]
live = evaluation["liveness"]

if not live["ok"]:
    st.error(
        f"**Liveness failed**: {live['errors']} error(s), {live['unparsed']} unparsed. "
        "A malformed-output bug does not move precision or recall — it removes pairs "
        "from the denominator. Fix this before reading anything below."
    )

if judge["n"] == 0:
    st.warning(
        f"No trusted labels cover this run (bank has {evaluation['n_labels']}). "
        "Nothing below is measured — label some pairs on the **Labelling** page."
    )

st.subheader("Funnel")
st.caption(
    "Where the volume went. Needs no labels — this is the first thing to read "
    "when a run produces less than expected."
)
try:
    flow = api.run_funnel(selected["id"])
    fcols = st.columns(2)
    for col, key, heading in (
        (fcols[0], "documents", "Documents"),
        (fcols[1], "pairs", "Pairs"),
    ):
        col.markdown(f"**{heading}**")
        for step in flow[key]:
            share = (
                f"{step['of_previous']:.1%} of previous"
                if step["of_previous"] is not None
                else "start"
            )
            col.markdown(f"`{step['count']:>7}`  {step['name']} — {share}")
            col.caption(step["note"])
    if flow["errored_verdicts"]:
        st.warning(
            f"{flow['errored_verdicts']} verdict(s) errored — they leave the "
            "denominator without moving any rate."
        )
    if flow["unjudged_selected"]:
        st.warning(
            f"{flow['unjudged_selected']} selected pair(s) were never judged — the "
            "run stopped early rather than the model failing."
        )
except ApiError as exc:
    st.error(exc.detail)

st.divider()
st.subheader("Detections")
st.caption(
    "What the pipeline found. Also needs no labels — but an unverified detection "
    "is a model's claim, not a fact."
)
try:
    found = api.run_detections(selected["id"])
    stats = found["summary"]
    dcols = st.columns(4)
    dcols[0].metric("Detections", stats["detections"])
    dcols[1].metric("Events", stats["events"], help="Distinct (concept, place, date).")
    dcols[2].metric("Confirmed", stats["by_verification"].get("confirmed", 0))
    dcols[3].metric(
        "Unverified",
        stats["by_verification"].get("unverified", 0),
        help="Nobody has reviewed these yet.",
    )

    if found["rows"]:
        # The rows are CSV-shaped, with "" for a missing number; a table needs None.
        shown = [
            {
                k: (None if v == "" and k in ("confidence", "vote_fraction") else v)
                for k, v in row.items()
            }
            for row in found["rows"]
        ]
        st.dataframe(shown, use_container_width=True)
        export_cols = st.columns(2)
        export_cols[0].download_button(
            "Download detections (CSV)",
            data=api.detections_csv(selected["id"]),
            file_name=f"detections-run{selected['id']}.csv",
            mime="text/csv",
        )
        export_cols[1].download_button(
            "Download events (CSV)",
            data=api.detections_csv(selected["id"], events=True),
            file_name=f"events-run{selected['id']}.csv",
            mime="text/csv",
        )
    else:
        st.caption("This run matched nothing.")
except ApiError as exc:
    st.error(exc.detail)

st.divider()
st.subheader("Judge")
st.caption(f"n = {judge['n']} labelled pairs")

cols = st.columns(4)
for col, name, ci_key in (
    (cols[0], "precision", "precision_ci"),
    (cols[1], "recall", "recall_ci"),
    (cols[2], "f1", "f1_ci"),
):
    value = judge[name]
    low, high = judge[ci_key]
    col.metric(name.title(), f"{value:.3f}" if value is not None else "—", help=describe(name))
    col.caption(f"[{low:.3f}, {high:.3f}]" if low is not None else "not measured")
cols[3].metric(
    "Confusion",
    f"{judge['tp']}/{judge['fp']}/{judge['tn']}/{judge['fn']}",
    help=describe("confusion"),
)
cols[3].caption("tp / fp / tn / fn")

if judge["n"]:
    st.altair_chart(charts.confusion_bar(judge), use_container_width=True)

st.divider()
st.subheader("Retrieval")
st.caption(
    "Scored separately from the judge on purpose: a concept never surfaced needs a "
    "different fix from one surfaced and then rejected."
)

cols = st.columns(4)
cols[0].metric(
    "Precision",
    f"{retrieval['precision']:.3f}" if retrieval["precision"] is not None else "—",
)
cols[1].metric(
    "Coverage",
    f"{retrieval['coverage']:.3f}" if retrieval["coverage"] is not None else "—",
)
cols[1].caption(describe("coverage"))
cols[2].metric("MRR", f"{retrieval['mrr']:.3f}" if retrieval["mrr"] is not None else "—")
cols[3].metric(
    "Cutoff recall",
    f"{retrieval['cutoff_recall']:.3f}" if retrieval["cutoff_recall"] is not None else "—",
)
cols[3].caption(describe("cutoff_recall"))

st.altair_chart(charts.recall_at_k(retrieval["recall_at_k"]), use_container_width=True)

with st.expander("Calibration — is the confidence number worth anything?"):
    st.caption(describe("calibration"))
    st.altair_chart(charts.calibration(evaluation["calibration"]), use_container_width=True)
    st.dataframe(
        [
            {
                "confidence": f"{b['range'][0]:.1f}–{b['range'][1]:.1f}",
                "n": b["n"],
                "accuracy": b["accuracy"],
            }
            for b in evaluation["calibration"]
        ],
        use_container_width=True,
    )

st.divider()
st.subheader("Breakdown")
st.caption(
    "Systematic failures only show up sliced: a pooled F1 cannot tell "
    '"uniformly mediocre" from "excellent except on one concept".'
)
dimension = st.radio("Slice by", ["concept", "category", "locus"], horizontal=True)
try:
    rows = api.run_breakdown(
        selected["id"], dimension=dimension, include_machine=include_machine
    )
    if rows:
        st.altair_chart(
            charts.metric_with_interval(rows, metric="f1", title=f"F1 by {dimension}"),
            use_container_width=True,
        )
        st.dataframe(
            [
                {
                    dimension: r["label"],
                    "n": r["n"],
                    "tp": r["tp"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                }
                for r in rows
            ],
            use_container_width=True,
        )
    else:
        st.caption("No labelled pairs to slice yet.")
except ApiError as exc:
    st.error(exc.detail)

st.divider()
st.subheader("Errors")
st.caption(
    "Every mistake with the model's own evidence attached. A confusion matrix "
    "says how many went wrong; this says why, which is where fixable patterns are."
)
try:
    result = api.run_errors(selected["id"], include_machine=include_machine)
    summary = result["summary"]
    error_cols = st.columns(3)
    error_cols[0].metric("Errors", summary["total"])
    error_cols[1].metric("False positives", summary["false_positives"])
    error_cols[2].metric("False negatives", summary["false_negatives"])

    for row in result["rows"][:25]:
        marker = "🔴 FP" if row["kind"] == "false_positive" else "🔵 FN"
        confidence = f"{row['confidence']:.2f}" if row["confidence"] is not None else "—"
        with st.expander(
            f"{marker}  {row['concept_name']}  ·  conf {confidence}  ·  "
            f"{(row['document_title'] or row['document_url'])[:60]}"
        ):
            if row["evidence"]:
                st.markdown(f"**Model quoted:** “{row['evidence']}”")
            if row["label_note"]:
                st.caption(f"Label note: {row['label_note']}")
            if row["reasoning"]:
                st.text_area(
                    "Reasoning trace",
                    row["reasoning"],
                    height=140,
                    key=f"reason_{row['document_id']}_{row['concept_id']}",
                )
            st.caption(f"[open article]({row['document_url']})")
except ApiError as exc:
    st.error(exc.detail)

st.divider()
st.subheader("Compare two runs")
st.caption(
    "Paired over the pairs both runs judged. Only the pairs they answered "
    "differently carry information about which is better."
)

pair = st.columns(2)
run_a = pair[0].selectbox("Run A", list(run_labels), key="cmp_a")
run_b = pair[1].selectbox(
    "Run B", list(run_labels), key="cmp_b", index=min(1, len(run_labels) - 1)
)

if st.button("Compare"):
    try:
        result = api.compare_runs(
            run_labels[run_a]["id"],
            run_labels[run_b]["id"],
            include_machine=include_machine,
        )
        st.info(result["verdict"])
        st.json(result["paired"], expanded=False)
    except ApiError as exc:
        st.error(exc.detail)

st.divider()
st.subheader("Leaderboard")
try:
    board = api.leaderboard()
    if board:
        st.altair_chart(charts.leaderboard_scatter(board), use_container_width=True)
        st.dataframe(board, use_container_width=True)
    else:
        st.caption("Nothing recorded yet — run `hontology eval run <id> --record`.")
except ApiError as exc:
    st.error(exc.detail)
