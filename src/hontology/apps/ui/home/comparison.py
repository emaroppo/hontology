"""Comparison: arms against a baseline, paired, as ``eval arms`` reports them."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.home.data import Context


def show_comparison(ctx: Context) -> None:
    labels = {
        r["run_id"]: f"{r['run_id']} · {r['name']} ({r['versions']['prompt_id']})"
        for r in ctx.runs
    }
    if len(labels) < 2:
        st.info("Comparing needs at least two judged runs.")
        return
    ids = list(labels)
    cols = st.columns(2)
    baseline = cols[0].selectbox(
        "Baseline", ids, format_func=lambda i: labels[i], key="cmp_baseline"
    )
    if baseline is None:
        return
    arms = cols[1].multiselect(
        "Arms",
        [i for i in ids if i != baseline],
        format_func=lambda i: labels[i],
        key=f"cmp_arms_{baseline}",
    )
    st.caption(
        "Each arm against the baseline on the same labelled articles, paired: an arm "
        "counts as an improvement only if the interval on its F1 difference lies above "
        "zero. The same tables `hontology eval arms` writes."
    )
    if not arms:
        return
    try:
        result = ctx.api.compare_arms(baseline, arms, ctx.manifest_path, ctx.annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.markdown(result["markdown"])
    st.download_button(
        "Download as Markdown",
        data=result["markdown"],
        file_name=f"arms-{baseline}-vs-{'-'.join(map(str, arms))}.md",
        mime="text/markdown",
    )
