"""Sample: a frozen random sample labelled a whole document at a time, blind.

Documents come in the sample's frozen order, so whatever prefix is done is
itself a random sample. Saving answers every leaf at once: the ticked ones
apply, the rest do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from hontology.apps.ui import panel
from hontology.apps.ui.client import Api, ApiError


def sample_tab(api: Api, ontologies: list[dict]) -> None:
    ontology = panel.ontology(api, ontologies)
    manifest_path = panel.current(api).sample
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read the sample manifest (Settings, top right): {exc}")
        return
    # A sample is labelled against the ontology of the run it was drawn from.
    try:
        drawn_from = api.get_run(manifest["run_id"])["ontology_id"]
    except (ApiError, KeyError):
        drawn_from = None
    if drawn_from is not None and drawn_from != ontology["id"]:
        st.warning(
            f"This sample was drawn from run {manifest['run_id']}, of another ontology. "
            "Pick that ontology in Settings (top right) to label it."
        )
        return
    first = st.number_input(
        "Label the first", min_value=1, value=120, step=10, key="sample_first"
    )

    document_ids = [entry["document_id"] for entry in manifest["order"]][: int(first)]
    try:
        statuses = api.documents_status(ontology["id"], document_ids)
        leaves = api.leaves(ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return

    done = sum(1 for s in statuses if s["labelled"])
    st.progress(done / len(statuses), text=f"{done} of {len(statuses)} labelled")

    # Start at the first document not yet labelled; navigation moves from there.
    if st.session_state.get("sample_manifest") != (manifest_path, int(first)):
        st.session_state["sample_manifest"] = (manifest_path, int(first))
        st.session_state.pop("sample_position", None)
    if "sample_position" not in st.session_state:
        st.session_state["sample_position"] = next(
            (i for i, s in enumerate(statuses) if not s["labelled"]), 0
        )
    position = _navigate(statuses, done)

    try:
        document = api.document_for_labelling(ontology["id"], document_ids[position])
    except ApiError as exc:
        st.error(exc.detail)
        return

    text_column, label_column = st.columns([3, 2])
    with text_column:
        _article(document, position, len(statuses))
    with label_column:
        _label_form(api, ontology["id"], document, leaves, statuses, position)


def _go(position: int, count: int) -> None:
    st.session_state["sample_position"] = max(0, min(position, count - 1))


def _navigate(statuses: list[dict], done: int) -> int:
    """Previous, next, next unlabelled and go to; returns the position shown."""
    count = len(statuses)
    position = st.session_state["sample_position"]
    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("← Previous", disabled=position == 0):
        _go(position - 1, count)
        st.rerun()
    if nav[1].button("Next →", disabled=position == count - 1):
        _go(position + 1, count)
        st.rerun()
    if nav[2].button("Next unlabelled", disabled=done == count):
        later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
        earlier = [i for i, s in enumerate(statuses) if not s["labelled"]]
        _go((later or earlier)[0], count)
        st.rerun()
    jump = nav[3].number_input(
        "Go to position", min_value=1, max_value=count, value=position + 1
    )
    if jump - 1 != position:
        _go(jump - 1, count)
        st.rerun()
    return position


def _article(document: dict, position: int, count: int) -> None:
    state = "labelled" if document["labelled"] else "not labelled yet"
    st.caption(f"Position {position + 1} of {count} · {state}")
    # Many scraped articles carry no title; the text's first line usually is one.
    first_line = (document["body"] or "").strip().split("\n", 1)[0][:160]
    st.subheader(document["title"] or first_line or document["url"])
    st.markdown(f"[{document['url']}]({document['url']})")
    with st.container(height=640):
        if document["body"]:
            st.text(document["body"])
        else:
            st.warning("No text stored for this article. Read it at the link above.")


def _label_form(
    api: Api,
    ontology_id: int,
    document: dict,
    leaves: list[dict],
    statuses: list[dict],
    position: int,
) -> None:
    """Every leaf as a checkbox, by family; saving moves to the next unlabelled."""
    st.markdown("**Which leaves does the article report?** Untick all for none.")
    current = set(document["positives"])
    key = document["document_id"]
    with st.form(f"labels_{key}"):
        chosen: list[int] = []
        family = None
        for leaf in leaves:
            if leaf["families"][0] != family:
                family = leaf["families"][0]
                st.markdown(f"**{family}**")
            explanation = "\n\n".join(
                part
                for part in (
                    leaf["definition"],
                    leaf["inclusion_criteria"] and f"Counts when: {leaf['inclusion_criteria']}",
                    leaf["exclusion_criteria"]
                    and f"Does not count when: {leaf['exclusion_criteria']}",
                )
                if part
            )
            if st.checkbox(
                leaf["name"],
                value=leaf["id"] in current,
                key=f"{key}_{leaf['id']}",
                help=explanation,
            ):
                chosen.append(leaf["id"])
        note = st.text_area("Note (optional)", value=document["note"] or "")
        saved = st.form_submit_button("Save and next", type="primary")

    if saved:
        try:
            api.label_document(ontology_id, key, chosen, note)
        except ApiError as exc:
            st.error(exc.detail)
            return
        later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
        _go(later[0] if later else position + 1, len(statuses))
        st.rerun()
