"""Health and export, for either kind of ontology."""

from __future__ import annotations

import json

import streamlit as st

from hontology.apps.ui.client import Api, ApiError


def show_health(api: Api, ontology_id: int) -> None:
    st.caption(
        "Checks that catch ontology bugs before they cost labelling hours. "
        "Advisory only — this is your ontology to author."
    )
    try:
        report = api.lint_ontology(ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    cols = st.columns(3)
    cols[0].metric("Classes", report["concepts"])
    cols[1].metric("Warnings", report["warnings"])
    cols[2].metric("Notes", report["info"])

    if not report["findings"]:
        st.success("Nothing flagged.")
    for finding in report["findings"]:
        renderer = st.warning if finding["severity"] == "warning" else st.info
        renderer(
            f"**{finding['concept_name']}** — {finding['check'].replace('_', ' ')}\n\n"
            f"{finding['message']}"
        )


def show_export(api: Api, ontology_id: int, slug: str) -> None:
    st.caption(
        "Keyed by class name throughout, so either file merges cleanly into another "
        "database and is safe to keep in version control next to your labels."
    )
    cols = st.columns(2)
    cols[0].download_button(
        "Download OWL (for Protégé)",
        data=api.export_owl(ontology_id),
        file_name=f"{slug}.ttl",
        mime="text/turtle",
    )
    exported = api.export_ontology(ontology_id)
    cols[1].download_button(
        "Download JSON",
        data=json.dumps(exported, indent=2, ensure_ascii=False),
        file_name=f"{slug}.json",
        mime="application/json",
    )
    st.json(exported, expanded=False)
