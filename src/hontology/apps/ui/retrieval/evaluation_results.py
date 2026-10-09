"""Evaluation results: recall at the cutoff, its cost in pairs sent to the judge,
recall at each depth of the ranking, and recall by class."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import charts
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.cutoff import describe_cutoff
from hontology.apps.ui.retrieval.context import Context


def show_results(
    ctx: Context, version: dict, cutoff: dict, result: dict, base: dict | None
) -> None:
    """*result* scored at *cutoff*; *base*, the loaded preset's, when the cutoff
    was edited from it, for the changes."""
    docs = result["documents"]
    if not result["positives"]:
        st.info(
            f"No labelled true match among the {docs['labelled']} labelled article(s) on "
            "this version's leaves. Try a machine annotation set as truth (Settings, "
            "top right)."
        )
        return
    _recall(result, base)
    _cost(ctx, version, cutoff)
    st.markdown("**Recall at each depth of the ranking**")
    st.caption(
        "Labelled true matches within the top k classes of their article: the ranking's "
        "quality before any cutoff. Where it flattens, a deeper cutoff stops paying."
    )
    st.altair_chart(charts.recall_at_k(result["recall_at_k"]), width="stretch")
    st.markdown("**By class**")
    st.dataframe(
        [
            {
                "Class": row["class"],
                "True matches": row["positives"],
                "In the pool": row["in_pool"],
                "Kept": row["kept"],
            }
            for row in result["per_class"]
        ],
        hide_index=True,
    )


def _interval(block: dict) -> str:
    low, high = block["ci"]
    return f"{block['rate']:.0%} ({low:.0%}–{high:.0%})" if low is not None else "—"


def _recall(result: dict, base: dict | None) -> None:
    docs = result["documents"]

    def change(now: float, then: float, fmt: str = "+d") -> str | None:
        return format(now - then, fmt) if base is not None and now != then else None

    cols = st.columns(5)
    cols[0].metric(
        "Recall at the cutoff",
        _interval(result["recall"]),
        delta=change(result["recall"]["found"], base["recall"]["found"]) if base else None,
        help="Labelled true matches kept, with a 95% Wilson interval. The arrow is the "
        "change in true matches kept, against the loaded preset.",
    )
    cols[1].metric("Recall in the pool", _interval(result["pool_recall"]))
    cols[2].metric(
        "Lost to the cutoff",
        result["lost_to_cutoff"],
        delta=change(result["lost_to_cutoff"], base["lost_to_cutoff"]) if base else None,
        delta_color="inverse",
    )
    cols[3].metric("Never ranked", result["never_ranked"])
    cols[4].metric(
        "Pairs per article",
        f"{result['pairs_per_document']:.2f}" if result["pairs_per_document"] else "—",
        delta=change(result["pairs_per_document"], base["pairs_per_document"], "+.2f")
        if base and result["pairs_per_document"] and base["pairs_per_document"]
        else None,
        delta_color="inverse",
        help="What the cutoff sends to the judge, on average, on the labelled articles.",
    )
    st.caption(
        f"{result['recall']['found']} of {result['positives']} labelled true matches kept, "
        f"over {docs['ranked']} labelled article(s)"
        + (
            f"; {docs['not_embedded']} not embedded under this version, so left out"
            if docs["not_embedded"]
            else ""
        )
        + f". Of the {result['kept_labelled']} kept labelled pairs, "
        f"{result['kept_positive']} are true matches."
    )


def _cost(ctx: Context, version: dict, cutoff: dict) -> None:
    """What the cutoff sends to the judge, over every article the version's
    ranking was built for (its source run's stored pool)."""
    source_id = version["source_run"]
    try:
        cost = ctx.api.retrieval_report(source_id, cutoff, ctx.annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    mine, theirs = cost["setting"], cost["run"]
    changed = cutoff != theirs["cutoff"]
    st.markdown(f"**Cost: pairs sent to the judge, on run {source_id}'s articles**")
    cols = st.columns(3)
    cols[0].metric(
        "Pairs kept",
        f"{mine['pairs_kept']:,}",
        delta=f"{mine['pairs_kept'] - theirs['pairs_kept']:+,}" if changed else None,
        delta_color="inverse",
    )
    cols[1].metric(
        "Per article",
        f"{mine['pairs_per_document']:.2f}",
        delta=f"{mine['pairs_per_document'] - theirs['pairs_per_document']:+.2f}"
        if changed
        else None,
        delta_color="inverse",
    )
    cols[2].metric(
        "Articles with none kept",
        f"{mine['documents_with_none']:,}",
        help="Never judged at all under this cutoff.",
    )
    st.caption(
        f"Over its {mine['documents']:,} articles"
        + (
            f", against its own cutoff ({describe_cutoff(theirs['cutoff'])})."
            if changed
            else ", which used this cutoff."
        )
    )
    histogram = {int(k): v for k, v in mine["kept_per_document"].items()}
    with st.expander("Classes kept per article", key="retrieval_histogram"):
        st.bar_chart(
            {"articles": [histogram.get(k, 0) for k in range(max(histogram, default=0) + 1)]},
            x_label="classes kept",
            y_label="articles",
        )
