"""Evaluation: a judge version across the runs that used it."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import charts, shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.judgement.context import Context


def show_evaluation(ctx: Context) -> None:
    try:
        judges = ctx.api.judgement_versions(ctx.ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not judges:
        st.info("No run of this ontology has judged anything yet.")
        return

    def describe(v: dict) -> str:
        return (
            f"{v['version']} · {v['prompt_id']} · {v['model']} · ontology "
            f"{v['ontology_version']} · {len(v['runs'])} run(s)"
        )

    by_version = {describe(v): v for v in judges}
    judge = by_version[st.selectbox("Judge version", list(by_version), key="ev_judge")]
    st.caption(f"Prompt text fingerprint {judge['prompt']}; provider {judge['provider']}.")
    run_labels = {f"{r['id']} · {r['name']} ({r['status']})": r["id"] for r in judge["runs"]}
    chosen = st.multiselect(
        "Runs pooled",
        list(run_labels),
        default=list(run_labels),
        key=f"ev_runs_{judge['version']}",
    )
    scope = st.radio(
        "Score on",
        ["responsible", "retrieved"],
        format_func=lambda s: {
            "responsible": "Everything this judge was responsible for",
            "retrieved": "Only pairs retrieval selected",
        }[s],
        horizontal=True,
        key="ev_scope",
        help="A per-pair or batched judge answers only what retrieval selected; a "
        "top-down judge answers for every leaf of the articles it processed, true "
        "matches retrieval never offered included. To compare judges of different "
        "styles, score them on what retrieval selected.",
    )
    if not chosen:
        return
    try:
        result = ctx.api.judgement_evaluate(
            [run_labels[c] for c in chosen], scope, ctx.annotator
        )
    except ApiError as exc:
        st.error(exc.detail)
        return
    _results(result)


def _results(result: dict) -> None:
    """The pooled scores, calibration and the per-run table."""
    pooled = result["pooled"]
    if not pooled["pairs"]:
        st.info(
            "No labelled pair among what these runs judged. Try a machine annotation set "
            "as truth (Settings, top right)."
        )
        return

    def interval(name: str) -> str:
        return shared.ci_text(pooled[name], pooled[f"{name}_ci"])

    cols = st.columns(4)
    cols[0].metric("Precision", interval("precision"))
    cols[1].metric("Recall", interval("recall"))
    cols[2].metric("F1", interval("f1"))
    cols[3].metric("Pairs scored", f"{pooled['pairs']:,}")
    st.caption(
        f"{pooled['tp']} true positives, {pooled['fp']} false positives, {pooled['fn']} "
        f"false negatives over {pooled['documents']} labelled article(s); 95% intervals "
        "from resampling articles. A pair several runs judged counts once, from the "
        "newest run."
    )
    with st.expander("Calibration: is the stated confidence worth anything?"):
        st.caption(
            "Agreement with the labels per confidence band, over the pairs the judge "
            "actually answered. Bars far from the diagonal mean the number is not a "
            "probability, however reasonable it looks."
        )
        st.altair_chart(charts.calibration(result["calibration"]), width="stretch")
    st.markdown("**Per run**")
    st.dataframe(
        [
            {
                "Run": f"{r['run_id']} · {r['name']}",
                "Pairs": r["pairs"],
                "Articles": r["documents"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "TP": r["tp"],
                "FP": r["fp"],
                "FN": r["fn"],
            }
            for r in result["runs"]
        ],
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
        },
    )
