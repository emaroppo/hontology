"""Codebooks: the public code lookups the links point into, loaded on request."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import ApiError
from hontology.apps.ui.filtering.context import SYSTEMS, Context


def show_codebooks(ctx: Context) -> None:
    ingest = {"cameo": ctx.api.ingest_cameo, "gkg-themes": ctx.api.ingest_themes}
    cols = st.columns(len(SYSTEMS))
    for col, (slug, title) in zip(cols, SYSTEMS.items(), strict=True):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            system = ctx.systems.get(slug)
            if system is None:
                st.caption("Not loaded.")
            else:
                n_codes = sum(level["n_codes"] for level in system["levels"])
                levels = ", ".join(level["level"] for level in system["levels"])
                st.caption(f"{n_codes:,} codes ({levels})")
            label = "Reload" if system else "Load"
            if st.button(label, key=f"ingest_{slug}", help="Fetches the public lookup."):
                try:
                    with st.spinner("Fetching…"):
                        result = ingest[slug]()
                    st.session_state["filtering_message"] = (
                        f"{title}: {result.get('total', '?'):,} codes loaded."
                    )
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
