"""Single run, by stage: a card per stage with a number or two, linking to the
stage's own page for its evaluation."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.home.data import Context, run_cutoff_report, run_funnel, run_sample
from hontology.apps.ui.shared import ci_text


def by_stage(ctx: Context, run_id: int, row: dict) -> dict:
    """The three stage cards; returns the run's sample scores, which the
    disagreements below are read from (empty when they could not be had)."""
    st.markdown("**By stage**")
    versions = row["versions"]
    cols = st.columns(3)
    with cols[0], st.container(border=True):
        _filtering(run_id, versions)
    with cols[1], st.container(border=True):
        _retrieval(ctx, run_id, versions)
    with cols[2], st.container(border=True):
        return _judgement(ctx, run_id, row)


def _filtering(run_id: int, versions: dict) -> None:
    st.markdown("Filtering")
    st.caption("Links " + (", ".join(versions["filter"]) or "not recorded for this run"))
    try:
        flow = run_funnel(run_id)
        steps = {s["name"]: s["count"] for s in flow["documents"]}
        st.metric("Articles this run considered", f"{steps.get('considered by this run', 0):,}")
    except ApiError as exc:
        st.error(exc.detail)
    st.page_link("views/2_Filtering.py", label="Filtering evaluation", icon="🧹")


def _retrieval(ctx: Context, run_id: int, versions: dict) -> None:
    st.markdown("Retrieval")
    source = versions["retrieval_source_run"]
    st.caption(
        f"Version {versions['retrieval']}"
        + (f", reused from run {source}" if source != run_id else "")
    )
    try:
        cut = run_cutoff_report(source, ctx.annotator)
        labelled = cut["labels"]
        st.metric(
            "True matches past the cutoff",
            f"{labelled['positives_kept']} of {labelled['positives']}"
            if labelled["positives"]
            else "—",
        )
        st.caption(f"{cut['pairs_per_document']:.2f} pairs per article to the judge")
    except ApiError as exc:
        st.error(exc.detail)
    st.page_link("views/3_Retrieval.py", label="Retrieval evaluation", icon="🔎")


def _judgement(ctx: Context, run_id: int, row: dict) -> dict:
    versions = row["versions"]
    st.markdown("Judgement")
    try:
        sample = run_sample(run_id, ctx.manifest_path, ctx.annotator)
    except ApiError as exc:
        sample = {}
        st.error(exc.detail)
    st.caption(f"Version {versions['judge']} · {versions['prompt_id']}")
    judge = (sample.get("article") or {}).get("judge_only") or {}
    st.metric(
        "Judge-only precision / recall",
        f"{ci_text(judge.get('precision'), None)} / {ci_text(judge.get('recall'), None)}",
        help="On the labelled pairs this run judged: what retrieval passed it.",
    )
    st.caption(f"{row['pairs_judged'] or 0} labelled pairs judged")
    st.page_link("views/4_Judgement.py", label="Judgement evaluation", icon="⚖️")
    return sample
