"""Ontology editor: create ontologies, author concepts, import and export.

This is the page that has to work from a completely empty install, because it is
the first thing anyone sees.
"""

from __future__ import annotations

import json

import streamlit as st

from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Ontology", page_icon="🧭", layout="wide")

api = Api()

st.title("🧭 Ontology")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()


# ---------------------------------------------------------------------------
# Empty state / selection
# ---------------------------------------------------------------------------

selected: dict | None = None

with st.sidebar:
    st.subheader("Ontology")
    if ontologies:
        labels = {f"{o['name']} ({o['slug']})": o for o in ontologies}
        selected = labels[st.selectbox("Select", list(labels))]
    else:
        st.caption("None yet — create one below.")

    with st.expander("Create new", expanded=not ontologies), st.form("create_ontology"):
        slug = st.text_input("Slug", placeholder="supply-chain")
        name = st.text_input("Name", placeholder="Supply chain disruption")
        description = st.text_area("Description", placeholder="Optional.")
        if st.form_submit_button("Create", type="primary"):
            try:
                api.create_ontology(
                    slug=slug.strip(), name=name.strip(), description=description or None
                )
                st.success(f"Created {name!r}.")
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

    with st.expander("Import from JSON"):
        uploaded = st.file_uploader("Export file", type="json")
        if uploaded is not None and st.button("Import"):
            try:
                api.import_ontology(json.loads(uploaded.read()))
                st.success("Imported. Re-importing merges rather than duplicating.")
                st.rerun()
            except (ApiError, json.JSONDecodeError) as exc:
                st.error(str(exc))

if selected is None:
    st.info(
        "**Nothing here yet.** Create an ontology in the sidebar to get started, "
        "or import a JSON export."
    )
    st.stop()

ontology_id = selected["id"]
concepts = api.list_concepts(ontology_id)


# ---------------------------------------------------------------------------
# Version banner
# ---------------------------------------------------------------------------

head = st.columns([3, 1, 1])
head[0].subheader(selected["name"])
head[1].metric("Concepts", len(concepts))
if head[2].button("Resolve version", help="Mint a version only if the wording changed"):
    ref = api.snapshot(ontology_id)
    if ref["created"]:
        st.toast(f"Minted {ref['version']} — wording changed.", icon="🆕")
    else:
        st.toast(f"Unchanged, still {ref['version']}.", icon="✅")

st.caption(
    "Editing a definition or its criteria changes what the model is asked, so it "
    "mints a new version and marks labels made under the old wording as stale. "
    "Changing a weight does not."
)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------

edit_tab, add_tab, export_tab = st.tabs(["Edit concepts", "Add concept", "Export"])

with edit_tab:
    if not concepts:
        st.info("No concepts yet. Add one in the next tab.")
    for concept in concepts:
        with st.expander(concept["name"]), st.form(f"edit_{concept['id']}"):
            name = st.text_input("Name", concept["name"])
            definition = st.text_area(
                "Definition",
                concept.get("definition") or "",
                help="What the concept means, in plain language.",
            )
            cols = st.columns(2)
            inclusion = cols[0].text_area(
                "Inclusion criteria",
                concept.get("inclusion_criteria") or "",
                help="What counts. Goes into the judge prompt verbatim.",
            )
            exclusion = cols[1].text_area(
                "Exclusion criteria",
                concept.get("exclusion_criteria") or "",
                help="What does not count — usually the best precision lever.",
            )
            weight = st.number_input(
                "Weight",
                value=float(concept.get("weight") or 0.0),
                step=0.5,
                help="Scoring weight. Does not affect versioning or labels.",
            )

            save, remove = st.columns(2)
            if save.form_submit_button("Save", type="primary"):
                try:
                    api.update_concept(
                        ontology_id,
                        concept["id"],
                        name=name.strip(),
                        definition=definition.strip(),
                        inclusion_criteria=inclusion.strip(),
                        exclusion_criteria=exclusion.strip(),
                        weight=weight,
                    )
                    st.success("Saved.")
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
            if remove.form_submit_button("Delete"):
                api.delete_concept(ontology_id, concept["id"])
                st.rerun()

with add_tab, st.form("add_concept"):
    name = st.text_input("Name", placeholder="Port closure")
    definition = st.text_area(
        "Definition", placeholder="A commercial seaport halts vessel operations."
    )
    cols = st.columns(2)
    inclusion = cols[0].text_area(
        "Inclusion criteria", placeholder="Closure has already taken effect."
    )
    exclusion = cols[1].text_area(
        "Exclusion criteria",
        placeholder="Threatened or announced closures; routine maintenance.",
    )
    category = st.text_input("Category", placeholder="logistics")
    weight = st.number_input("Weight", value=1.0, step=0.5)

    if st.form_submit_button("Add concept", type="primary"):
        try:
            api.create_concept(
                ontology_id,
                name=name.strip(),
                definition=definition.strip() or None,
                inclusion_criteria=inclusion.strip() or None,
                exclusion_criteria=exclusion.strip() or None,
                category=category.strip() or None,
                weight=weight,
            )
            st.success(f"Added {name!r}.")
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)

with export_tab:
    exported = api.export_ontology(ontology_id)
    st.caption(
        "Keyed by name throughout, so this file merges cleanly into another "
        "database and is safe to keep in version control next to your labels."
    )
    st.download_button(
        "Download JSON",
        data=json.dumps(exported, indent=2, ensure_ascii=False),
        file_name=f"{selected['slug']}.json",
        mime="application/json",
    )
    st.json(exported, expanded=False)
