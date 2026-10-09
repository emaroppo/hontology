"""What it keeps: what the filter would do to the articles not downloaded yet.
Slow, so computed on request and kept for the session."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.filtering.context import Context


def cached(ctx: Context, kind: str) -> dict | None:
    return st.session_state.get("filter_cache", {}).get((ctx.ontology_id, kind))


def compute(ctx: Context, kind: str, fetch) -> None:
    try:
        with st.spinner("Walking every feed record; this takes a few minutes…"):
            data = fetch(ctx.ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.session_state.setdefault("filter_cache", {})[(ctx.ontology_id, kind)] = {
        "data": data,
        "at": datetime.now().strftime("%H:%M"),
        "fingerprint": ctx.fingerprint,
    }
    st.rerun()


def freshness(ctx: Context, entry: dict) -> None:
    if entry["fingerprint"] != ctx.fingerprint:
        st.warning(f"Computed at {entry['at']}, before the links last changed. Recompute.")
    else:
        st.caption(f"Computed at {entry['at']}.")


def show_preview(ctx: Context) -> None:
    st.caption(
        "What the filter would do to the articles not downloaded yet, which is "
        "what the download budget is spent on. Nothing is fetched."
    )
    entry = cached(ctx, "preview")
    if st.button("Recompute" if entry else "Compute", key="preview_go", type="primary"):
        compute(ctx, "preview", ctx.api.filter_preview)
    if entry is None:
        return
    freshness(ctx, entry)
    data = entry["data"]
    if not data["usable"]:
        st.info(data["reason"])
        return
    cols = st.columns(4)
    cols[0].metric("Share kept", f"{data['share_kept']:.1%}")
    cols[1].metric("Would download", f"{data['unfetched_matching']:,}")
    cols[2].metric("Would skip", f"{data['unfetched_skipped']:,}")
    cols[3].metric("Linked codes", data["linked_codes"])
    st.caption(
        f"Of {data['documents_total']:,} feed articles, {data['documents_matching']:,} "
        f"match some link; {data['documents_unfetched']:,} are not downloaded yet."
    )
