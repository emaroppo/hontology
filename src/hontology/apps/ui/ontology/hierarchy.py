"""A structured ontology: its tree, and one class in detail.

The structure, names and internal classes are authored in an OWL editor and
arrive by import; only leaf wording is edited here.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import Api
from hontology.apps.ui.ontology.wording import wording_form


def show_hierarchy(api: Api, ontology_id: int, classes: list[dict]) -> None:
    order = shared.tree_order(classes)
    left, right = st.columns([2, 3])

    with left, st.container(height=640):
        lines = []
        for depth, concept in order:
            name = concept["name"] if concept["leaf"] else f"**{concept['name']}**"
            if len(concept["parents"]) > 1:
                name += " ⧉"
            lines.append(f"{'  ' * depth}- {name}")
        st.markdown("\n".join(lines))
        st.caption("**Bold**: internal class. ⧉: has more than one parent (either, not both).")

    with right:
        first_seen: dict[int, str] = {}
        for depth, concept in order:
            first_seen.setdefault(concept["id"], f"{'· ' * depth}{concept['name']}")
        options = {label: concept_id for concept_id, label in first_seen.items()}
        by_id = {c["id"]: c for c in classes}
        concept = by_id[options[st.selectbox("Class", list(options))]]
        _class_detail(api, ontology_id, classes, concept)


def _class_detail(api: Api, ontology_id: int, classes: list[dict], concept: dict) -> None:
    kind = "Leaf" if concept["leaf"] else "Internal class"
    st.markdown(f"### {concept['name']}")
    st.caption(
        " · ".join(
            part
            for part in (
                kind,
                f"category: {concept['category']}" if concept["category"] else None,
                f"weight {concept['weight']:g}" if concept["weight"] is not None else None,
            )
            if part
        )
    )

    facts = {
        "Parents": concept["parents"],
        "Children": concept["children"],
        "Precursor of": concept["precursor_of"],
        "Preceded by": sorted(
            c["name"] for c in classes if concept["name"] in c["precursor_of"]
        ),
        "Groups": concept["groups"],
    }
    for label, values in facts.items():
        if values:
            st.markdown(f"**{label}:** {', '.join(values)}")

    if concept["leaf"]:
        wording_form(api, ontology_id, concept, key=f"wording_{concept['id']}")
        return
    for field, label in (
        ("definition", "Definition"),
        ("inclusion_criteria", "Inclusion criteria"),
        ("exclusion_criteria", "Exclusion criteria"),
    ):
        if concept.get(field):
            st.markdown(f"**{label}.** {concept[field]}")
    st.info(
        "Internal classes, names, categories and the structure itself are "
        "edited in Protégé: download the OWL file from **Export**, edit it, "
        "and import it above. Leaf wording can be edited here."
    )
