"""Try a cutoff: the ranking for one thing at a time, as the run chosen in
Settings ranks it, through a cutoff that starts from the run's own."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.cutoff import (
    CUT_KEYS,
    cutoff_sliders,
    describe_cutoff,
    read_cutoff,
    seed_cutoff,
)
from hontology.apps.ui.retrieval.context import Context
from hontology.apps.ui.retrieval.try_views import show_article, show_class, show_own_text


def cutoff_controls(run: dict) -> dict:
    """The cutoff Try applies, in a collapsible section, starting from the run's own."""
    own = run["cutoff"]
    # Kept across a visit to Evaluation, which leaves these widgets undrawn.
    shared.restore(CUT_KEYS)
    if st.session_state.get("retrieval_cutoff_for") != run["id"]:
        st.session_state["retrieval_cutoff_for"] = run["id"]
        seed_cutoff("cut_", own)
    current = read_cutoff("cut_")
    edited = current != {k: own[k] for k in current}
    state = "edited" if edited else f"run {run['id']}'s own"
    header = f"**Cutoff** · {describe_cutoff(current)} · {state}"
    with st.expander(header, key="retrieval_cutoff_panel"):
        cutoff_sliders(st.columns(5), "cut_", helps=True)
        cols = st.columns([1, 4])
        if cols[0].button("Back to the run's own", disabled=not edited):
            st.session_state.pop("retrieval_cutoff_for", None)
            for key in CUT_KEYS:
                st.session_state.pop(f"keep:{key}", None)
            st.rerun()
        cols[1].caption(
            f"Run {run['id']}'s own: {describe_cutoff(own)}. Its pool holds the top 20 "
            "classes per article; a cutoff can only draw from that."
        )
    shared.persist(CUT_KEYS)
    return read_cutoff("cut_")


def no_run(run: dict | None) -> bool:
    if run is None:
        st.info(
            "Pick a run that has retrieved something in Settings (top right): Tune and "
            "Explore "
            "work on a run's stored ranking."
        )
        return True
    return False


def show_try(ctx: Context) -> None:
    run = ctx.run
    if no_run(run):
        return
    assert run is not None
    cols = st.columns([3, 2])
    look = cols[1].radio(
        "Look at",
        ["Your text", "A corpus article", "A class's articles"],
        horizontal=True,
        key="retrieval_look",
        help="Your text: an article pasted in, ranked as a corpus article would be. "
        "It is shared with Judgement's Your text.",
    )
    document_id = None
    leaf = None
    if look == "A corpus article":
        try:
            labelled = ctx.api.retrieval_labelled(run["id"], ctx.annotator)
        except ApiError as exc:
            st.error(exc.detail)
            return
        options = {
            d["document_id"]: f"{d['title'] or d['url'][:90]}"
            + (f"  ({d['positives']} match)" if d["positives"] == 1 else "")
            + (f"  ({d['positives']} matches)" if d["positives"] > 1 else "")
            for d in labelled
        }
        if not options:
            cols[0].info("No labelled article in this run.")
            return
        document_id = cols[0].selectbox(
            f"Labelled article ({len(options)})",
            list(options),
            format_func=lambda i: options[i],
            key="retrieval_article",
        )
    elif look == "A class's articles":
        try:
            leaves = ctx.api.leaves(ctx.ontology["id"])
        except ApiError as exc:
            st.error(exc.detail)
            return
        by_name = shared.leaf_options(leaves)
        leaf = by_name[cols[0].selectbox("Class", list(by_name), key="retrieval_class")]
    else:
        cols[0].caption(
            f"Ranked as run {run['id']}'s retrieval ranks an article: its embedding "
            "model, class wording and article length."
        )

    cutoff = cutoff_controls(run)
    if look == "Your text":
        show_own_text(ctx, cutoff)
    elif document_id is not None:
        show_article(ctx, document_id, cutoff)
    elif leaf is not None:
        show_class(ctx, leaf, cutoff)
