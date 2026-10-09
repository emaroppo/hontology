"""A class's wording, definition and criteria: the part of it the judge reads."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import Api


def wording_fields(concept: dict) -> dict[str, str]:
    """Definition and criteria, the part of a class the judge reads, as text areas
    in the form being drawn. Returns them stripped, by field."""
    definition = st.text_area(
        "Definition",
        concept.get("definition") or "",
        help="What the class means, in plain language.",
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
    return {
        "definition": definition.strip(),
        "inclusion_criteria": inclusion.strip(),
        "exclusion_criteria": exclusion.strip(),
    }


def wording_form(api: Api, ontology_id: int, concept: dict, *, key: str) -> None:
    """The wording alone, in a form of its own."""
    with st.form(key):
        wording = wording_fields(concept)
        if st.form_submit_button("Save wording", type="primary"):
            shared.act(
                api.update_concept, ontology_id, concept["id"], success="Saved.", **wording
            )
