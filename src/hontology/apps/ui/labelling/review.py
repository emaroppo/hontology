"""The bank's counts, and the review that makes a label count: machine proposals
awaiting a human, and labels gone stale after a concept was reworded."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import Api, ApiError


def bank_stats(stats: dict) -> None:
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


def review(api: Api, ontology_id: int, stats: dict) -> None:
    """Proposals awaiting review, then stale labels, each when there are any."""
    if stats["pending_adjudication"]:
        st.subheader(f"Awaiting review ({stats['pending_adjudication']})")
        st.caption(
            "Machine proposals. They do **not** count as ground truth until confirmed "
            "or flipped here — scoring a model against another model's unreviewed "
            "labels measures agreement, not correctness. Disagreeing is the most "
            "valuable outcome, not an error."
        )
        try:
            _adjudication_panel(
                api, api.pending_adjudication(ontology_id, limit=25), kind="pending"
            )
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
                _adjudication_panel(api, api.stale_detail(ontology_id, limit=25), kind="stale")
            except ApiError as exc:
                st.error(exc.detail)
        st.divider()


def _adjudication_panel(api: Api, rows: list[dict], *, kind: str) -> None:
    """Confirm or flip a proposal. This is the step that makes a label count."""
    for row in rows:
        with st.container(border=True):
            head = st.columns([3, 1])
            title = row["document_title"] or row["document_url"][:64]
            head[0].markdown(f"**{row['concept_name']}** · [{title}]({row['document_url']})")
            proposed = "matched" if row["proposed_matched"] else "no match"
            by = row.get("proposed_by") or "a model"
            head[1].markdown(f"proposed: **{proposed}**")
            head[1].caption(f"by {by} · {row.get('ontology_version') or '—'}")

            if row.get("note"):
                st.caption(f"note: {row['note']}")

            buttons = st.columns([1, 1, 6])
            key = f"{kind}_{row['id']}"
            for col, label, prefix, matched, primary in (
                (buttons[0], "Confirm", "ok", row["proposed_matched"], True),
                (buttons[1], "Flip", "flip", not row["proposed_matched"], False),
            ):
                if col.button(
                    label, key=f"{prefix}_{key}", type="primary" if primary else "secondary"
                ):
                    shared.act(api.adjudicate_label, row["id"], matched=matched)
