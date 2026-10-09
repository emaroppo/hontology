"""A flat ontology, a list of event classes, authored entirely here: each class
edited in place, and new ones added several at once or one in full."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import Api, ApiError
from hontology.apps.ui.ontology.wording import wording_fields


def show_flat_editor(api: Api, ontology_id: int, classes: list[dict]) -> None:
    if not classes:
        st.info("No classes yet. Add some in the next tab.")
    for concept in classes:
        with st.expander(concept["name"]), st.form(f"edit_{concept['id']}"):
            name = st.text_input("Name", concept["name"])
            wording = wording_fields(concept)
            cols = st.columns(2)
            category = cols[0].text_input("Category", concept.get("category") or "")
            weight = cols[1].number_input(
                "Weight",
                value=float(concept.get("weight") or 0.0),
                step=0.5,
                help="Scoring weight. Does not affect versioning or labels.",
            )

            save, remove = st.columns(2)
            if save.form_submit_button("Save", type="primary"):
                shared.act(
                    api.update_concept,
                    ontology_id,
                    concept["id"],
                    success="Saved.",
                    name=name.strip(),
                    **wording,
                    category=category.strip() or None,
                    weight=weight,
                )
            if remove.form_submit_button("Delete"):
                api.delete_concept(ontology_id, concept["id"])
                st.rerun()


def show_add(api: Api, ontology_id: int) -> None:
    st.markdown("**Several at once**")
    with st.form("add_many"):
        text = st.text_area(
            "One class per line, as `Name: definition`",
            placeholder=(
                "Port closure: A commercial seaport halts vessel operations.\n"
                "Rail strike: Rail freight workers stop work."
            ),
            height=160,
        )
        if st.form_submit_button("Add all", type="primary"):
            added, errors = 0, []
            for line in text.splitlines():
                if not line.strip():
                    continue
                name, _, definition = line.partition(":")
                try:
                    api.create_concept(
                        ontology_id,
                        name=name.strip(),
                        definition=definition.strip() or None,
                        weight=1.0,
                    )
                    added += 1
                except ApiError as exc:
                    errors.append(f"{name.strip()!r}: {exc.detail}")
            for error in errors:
                st.error(error)
            if added:
                st.success(f"Added {added} class(es). Add criteria to sharpen them.")
                if not errors:
                    st.rerun()

    st.markdown("**One, in full**")
    with st.form("add_concept"):
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
        cols = st.columns(2)
        category = cols[0].text_input("Category", placeholder="logistics")
        weight = cols[1].number_input("Weight", value=1.0, step=0.5)

        if st.form_submit_button("Add class", type="primary"):
            shared.act(
                api.create_concept,
                ontology_id,
                success=f"Added {name!r}.",
                name=name.strip(),
                definition=definition.strip() or None,
                inclusion_criteria=inclusion.strip() or None,
                exclusion_criteria=exclusion.strip() or None,
                category=category.strip() or None,
                weight=weight,
            )
