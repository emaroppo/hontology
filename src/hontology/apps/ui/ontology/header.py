"""New and imported ontologies, and the header of the one shown: counts and
versions."""

from __future__ import annotations

import json

import streamlit as st

from hontology.apps.ui import panel
from hontology.apps.ui.client import Api, ApiError


def new_and_import(api: Api, ontologies: list[dict]) -> None:
    """Create an ontology, or import one; either becomes the one shown."""
    new_col, import_col = st.columns(2)
    with new_col.expander("New ontology", expanded=not ontologies), st.form("create_ontology"):
        new_slug = st.text_input("Slug", placeholder="supply-chain")
        new_name = st.text_input("Name", placeholder="Supply chain disruption")
        description = st.text_area("Description", placeholder="Optional.")
        if st.form_submit_button("Create", type="primary"):
            try:
                created = api.create_ontology(
                    slug=new_slug.strip(),
                    name=new_name.strip(),
                    description=description or None,
                )
                panel.request("ontology", created["id"])
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

    with import_col.expander("Import", expanded=not ontologies):
        st.caption(
            "OWL (Turtle, `.ttl`) from Protégé, or a JSON export. Re-importing "
            "merges by class name rather than duplicating."
        )
        uploaded = st.file_uploader("File", type=["ttl", "json"])
        reword = st.checkbox(
            "Allow rewording existing classes",
            help="OWL only. Off by default, so a structural edit cannot silently "
            "change the wording labels were made against.",
        )
        if uploaded is not None and st.button("Import"):
            try:
                raw = uploaded.read().decode("utf-8")
                if uploaded.name.endswith(".ttl"):
                    imported = api.import_owl(raw, allow_text_change=reword)
                else:
                    imported = api.import_ontology(json.loads(raw))
                panel.request("ontology", imported["id"])
                st.rerun()
            except (ApiError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(str(exc))


def version_header(api: Api, selected: dict, classes: list[dict], structured: bool) -> None:
    """The ontology's name and counts, and where its live wording stands."""
    listing = api.versions(selected["id"])
    versions: list[dict] = listing["versions"]
    current: str | None = listing["current"]
    latest = versions[-1]["version"] if versions else None

    head = st.columns([3, 1, 1, 1])
    head[0].subheader(selected["name"])
    head[1].metric("Classes", len(classes))
    if structured:
        head[2].metric("Leaves", sum(c["leaf"] for c in classes))
    head[3].metric("Version", current or (f"{latest} + edits" if latest else "—"))

    # Versions are minted automatically by whatever needs one, so there is nothing
    # to press here: the header only says where the live wording stands.
    if current is None and classes:
        upcoming = f"v{len(versions) + 1}"
        status = (
            f"Edited since {latest}. The edits become **{upcoming}** automatically "
            "when the next run starts or the next label is saved."
            if latest
            else f"Not versioned yet. It becomes **{upcoming}** automatically when "
            "the first run starts or the first label is saved."
        )
        st.info(status)

    caption = (
        "A version pins the wording the model is asked about. Editing a definition or "
        "its criteria (or, in a hierarchy, the structure) makes the next one, and "
        "labels made under the old wording are marked stale. Weights and categories "
        "do not."
    )
    if versions:
        history = " · ".join(
            f"{v['version']} ({v['n_concepts']} classes, {v['created_at'][:10]})"
            for v in versions
        )
        caption += f"\n\nVersions: {history}"
    st.caption(caption)
