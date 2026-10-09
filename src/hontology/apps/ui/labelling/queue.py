"""Queue: single pairs, ranked by how much a label would teach.

It has to answer two questions fast: *why is this pair in front of me*, and
*what did the models say*. Both are shown, because a queue that hides its
reasoning trains you to rubber stamp it.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import panel, shared
from hontology.apps.ui.client import Api, ApiError
from hontology.apps.ui.labelling.bank import move_the_bank
from hontology.apps.ui.labelling.review import bank_stats, review

REASON_HELP = {
    "runs disagree": "Two runs split on this pair, so at least one is wrong. "
    "This needs no model to be calibrated, which is why it ranks first.",
    "model uncertain": "A run whose confidence carries real signal sat near the "
    "decision boundary here.",
    "never judged": "Retrieved but never judged by any run — a blind spot nothing "
    "else surfaces.",
    "coverage": "This concept has few labels relative to the others.",
}


def queue_tab(api: Api, ontologies: list[dict]) -> None:
    ontology = panel.ontology(api, ontologies)
    cols = st.columns(3)
    limit = cols[0].slider("Queue size", 5, 100, 25, 5)
    per_concept_cap = cols[1].number_input(
        "Max per concept",
        min_value=0,
        value=5,
        help="0 means no cap. A cap stops the noisiest concept from filling the queue.",
    )
    include_unjudged = cols[2].toggle(
        "Include never-judged pairs",
        value=True,
        help="Retrieved but never judged. Nothing else surfaces these.",
    )

    move_the_bank(api, ontology)
    stats = api.label_stats(ontology["id"])
    bank_stats(stats)
    st.divider()
    review(api, ontology["id"], stats)

    st.subheader("Queue")

    try:
        items = api.labelling_queue(
            ontology["id"],
            limit=limit,
            per_concept_cap=int(per_concept_cap) or None,
            include_unjudged=include_unjudged,
        )
    except ApiError as exc:
        st.error(exc.detail)
        return

    if not items:
        st.success(
            "Nothing queued. Either everything retrieved has been labelled, or no runs "
            "have been executed yet."
        )
        return

    st.caption(f"{len(items)} pair(s) queued, most informative first.")
    for item in items:
        _pair(api, item)


def _pair(api: Api, item: dict) -> None:
    """One queued pair: why it is here, what the runs said, and the answer."""
    with st.container(border=True):
        head = st.columns([3, 1])
        title = item["document_title"] or item["document_url"][:70]
        head[0].markdown(f"**{item['concept_name']}** · [{title}]({item['document_url']})")
        head[1].markdown(f"`{item['reason']}`")
        head[1].caption(REASON_HELP.get(item["reason"], ""))

        if item["verdicts"]:
            for verdict in item["verdicts"]:
                _verdict(verdict)
        else:
            st.caption("No run has judged this pair.")

        note = st.text_input(
            "Note (optional)", key=f"note_{item['document_id']}_{item['concept_id']}"
        )
        buttons = st.columns([1, 1, 6])
        key = f"{item['document_id']}_{item['concept_id']}"
        for col, label, prefix, matched, primary in (
            (buttons[0], "Matches", "yes", True, True),
            (buttons[1], "Does not", "no", False, False),
        ):
            if col.button(
                label, key=f"{prefix}_{key}", type="primary" if primary else "secondary"
            ):
                shared.act(
                    api.create_label,
                    document_id=item["document_id"],
                    concept_id=item["concept_id"],
                    matched=matched,
                    note=note or None,
                )


def _verdict(verdict: dict) -> None:
    mark = "matched" if verdict["matched"] else "no match"
    confidence = f"{verdict['confidence']:.2f}" if verdict["confidence"] is not None else "—"
    trust = "" if verdict["confidence_usable"] else "  ·  confidence not informative"
    st.markdown(
        f"&nbsp;&nbsp;run {verdict['run_id']}: **{mark}** (confidence {confidence}){trust}",
        unsafe_allow_html=True,
    )
    if verdict.get("evidence"):
        st.caption(f"“{verdict['evidence'][:220]}”")
