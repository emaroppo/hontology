"""The labelling queue.

The screen where ground truth actually gets made, so it has to answer two
questions fast: *why is this pair in front of me*, and *what did the models say*.
Both are shown, because a queue that hides its reasoning trains you to rubber
stamp it — and a rubber-stamped machine proposal is exactly the label that
poisons the metrics it is supposed to validate.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Labelling", page_icon="🏷️", layout="wide")

api = Api()

st.title("🏷️ Labelling")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

with st.sidebar:
    labels_by_name = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    ontology = labels_by_name[st.selectbox("Ontology", list(labels_by_name))]
    limit = st.slider("Queue size", 5, 100, 25, 5)
    per_concept_cap = st.number_input(
        "Max per concept",
        min_value=0,
        value=5,
        help="0 means no cap. A cap stops the noisiest concept from filling the queue.",
    )
    include_unjudged = st.toggle(
        "Include never-judged pairs",
        value=True,
        help="Retrieved but never judged. Nothing else surfaces these.",
    )

stats = api.label_stats(ontology["id"])

row = st.columns(5)
row[0].metric("Labels", stats["total"])
row[1].metric("Counted", stats["trusted"], help="Human, adjudicated or imported.")
row[2].metric(
    "Awaiting review",
    stats["pending_adjudication"],
    help="Machine proposals. These do NOT count until a human confirms them.",
)
row[3].metric(
    "Stale",
    stats["stale"],
    help="The concept was reworded after the label was made, so it answers a "
    "question no longer being asked. Excluded from metrics by default.",
)
row[4].metric("Observations", stats.get("observations", 0))

if stats["stale"]:
    st.warning(
        f"{stats['stale']} label(s) went stale after a concept was edited. They are "
        "excluded from metrics until re-adjudicated — confirming one against the "
        "current wording makes it count again."
    )

st.divider()


def adjudication_panel(rows: list[dict], *, kind: str) -> None:
    """Confirm or flip a proposal. This is the step that makes a label count."""
    for row in rows:
        with st.container(border=True):
            head = st.columns([3, 1])
            head[0].markdown(
                f"**{row['concept_name']}** · "
                f"[{row['document_title'] or row['document_url'][:64]}]({row['document_url']})"
            )
            proposed = "matched" if row["proposed_matched"] else "no match"
            by = row.get("proposed_by") or "a model"
            head[1].markdown(f"proposed: **{proposed}**")
            head[1].caption(f"by {by} · {row.get('ontology_version') or '—'}")

            if row.get("note"):
                st.caption(f"note: {row['note']}")

            buttons = st.columns([1, 1, 6])
            key = f"{kind}_{row['id']}"
            if buttons[0].button("Confirm", key=f"ok_{key}", type="primary"):
                try:
                    api.adjudicate_label(row["id"], matched=row["proposed_matched"])
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
            if buttons[1].button("Flip", key=f"flip_{key}"):
                try:
                    api.adjudicate_label(row["id"], matched=not row["proposed_matched"])
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)


if stats["pending_adjudication"]:
    st.subheader(f"Awaiting review ({stats['pending_adjudication']})")
    st.caption(
        "Machine proposals. They do **not** count as ground truth until confirmed "
        "or flipped here — scoring a model against another model's unreviewed "
        "labels measures agreement, not correctness. Disagreeing is the most "
        "valuable outcome, not an error."
    )
    try:
        adjudication_panel(api.pending_adjudication(ontology["id"], limit=25), kind="pending")
    except ApiError as exc:
        st.error(exc.detail)
    st.divider()

if stats["stale"]:
    with st.expander(f"Stale labels ({stats['stale']}) — re-adjudicate to restore them"):
        st.caption(
            "The concept was reworded after these were made, so they answer a "
            "question no longer being asked. Confirming against the current wording "
            "makes them count again."
        )
        try:
            adjudication_panel(api.stale_detail(ontology["id"], limit=25), kind="stale")
        except ApiError as exc:
            st.error(exc.detail)
    st.divider()

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
    st.stop()

if not items:
    st.success(
        "Nothing queued. Either everything retrieved has been labelled, or no runs "
        "have been executed yet."
    )
    st.stop()

REASON_HELP = {
    "runs disagree": "Two runs split on this pair, so at least one is wrong. "
    "This needs no model to be calibrated, which is why it ranks first.",
    "model uncertain": "A run whose confidence carries real signal sat near the "
    "decision boundary here.",
    "never judged": "Retrieved but never judged by any run — a blind spot nothing "
    "else surfaces.",
    "coverage": "This concept has few labels relative to the others.",
}

st.caption(f"{len(items)} pair(s) queued, most informative first.")

for item in items:
    with st.container(border=True):
        head = st.columns([3, 1])
        head[0].markdown(
            f"**{item['concept_name']}** · "
            f"[{item['document_title'] or item['document_url'][:70]}]({item['document_url']})"
        )
        head[1].markdown(f"`{item['reason']}`")
        head[1].caption(REASON_HELP.get(item["reason"], ""))

        if item["verdicts"]:
            for verdict in item["verdicts"]:
                mark = "matched" if verdict["matched"] else "no match"
                confidence = (
                    f"{verdict['confidence']:.2f}" if verdict["confidence"] is not None else "—"
                )
                trust = (
                    "" if verdict["confidence_usable"] else "  ·  confidence not informative"
                )
                st.markdown(
                    f"&nbsp;&nbsp;run {verdict['run_id']}: **{mark}** "
                    f"(confidence {confidence}){trust}",
                    unsafe_allow_html=True,
                )
                if verdict.get("evidence"):
                    st.caption(f"“{verdict['evidence'][:220]}”")
        else:
            st.caption("No run has judged this pair.")

        note = st.text_input(
            "Note (optional)", key=f"note_{item['document_id']}_{item['concept_id']}"
        )
        buttons = st.columns([1, 1, 6])
        key = f"{item['document_id']}_{item['concept_id']}"

        if buttons[0].button("Matches", key=f"yes_{key}", type="primary"):
            try:
                api.create_label(
                    document_id=item["document_id"],
                    concept_id=item["concept_id"],
                    matched=True,
                    note=note or None,
                )
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

        if buttons[1].button("Does not", key=f"no_{key}"):
            try:
                api.create_label(
                    document_id=item["document_id"],
                    concept_id=item["concept_id"],
                    matched=False,
                    note=note or None,
                )
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
