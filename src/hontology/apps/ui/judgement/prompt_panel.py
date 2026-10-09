"""The prompt a trial asks with, in one collapsible section: its template, system
text and the class's wording, editable, and the wording savable to the class."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.judgement.context import WORDING, Context


def prompt_panel(ctx: Context, leaf: dict) -> tuple[str, str, dict, bool]:
    """The prompt in one collapsible section: its template, system text and the
    class's wording, editable. Returns the prompt id, system text and wording as
    they stand, and whether either was edited."""
    run, templates = ctx.run, ctx.templates
    assert run is not None
    prompt_ids = list(templates)
    prompt_key = f"judgement_prompt_{run['id']}"
    # The template and the edits are widget values, which Streamlit drops when the
    # page or tab that drew them is left: keep a copy, so they survive a visit
    # elsewhere and are never read missing.
    kept_keys = [prompt_key, "edit_system", *(f"edit_{field}" for field in WORDING)]
    shared.restore(kept_keys)
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = (
            run["prompt_id"] if run["prompt_id"] in templates else prompt_ids[0]
        )
    prompt_id = st.session_state[prompt_key]
    # Edits start from the template and the class as saved; a new class, prompt or
    # run starts over.
    context = (leaf["id"], prompt_id)
    missing = any(key not in st.session_state for key in kept_keys)
    if st.session_state.get("judgement_context") != context or missing:
        st.session_state["judgement_context"] = context
        st.session_state["edit_system"] = templates[prompt_id]["system"]
        for field in WORDING:
            st.session_state[f"edit_{field}"] = leaf.get(field) or ""

    system = st.session_state["edit_system"]
    wording = {field: st.session_state[f"edit_{field}"].strip() or None for field in WORDING}
    original_wording = {field: (leaf.get(field) or "").strip() or None for field in WORDING}
    system_edited = system != templates[prompt_id]["system"]
    wording_edited = wording != original_wording
    edited = system_edited or wording_edited
    changes = [
        part
        for part, changed in (("system text", system_edited), ("class wording", wording_edited))
        if changed
    ]
    state = f"edited: {' and '.join(changes)}" if changes else "as saved"

    with st.expander(f"**Prompt** · {prompt_id} · {state}", key="judgement_prompt_panel"):
        cols = st.columns([2, 3])
        cols[0].selectbox("Template", prompt_ids, key=prompt_key)
        cols[1].caption(
            f"Asked as run {run['id']}: {run['provider']} · {run['model']}, with its "
            "decoding settings and article text limit."
            + (
                ""
                if run["prompt_id"] in templates
                else f" It used {run['prompt_id']}, which is not a per-pair prompt; its "
                "verdicts are shown for reference, and the trial asks per pair."
            )
        )
        st.text_area("System text", key="edit_system", height=260)
        st.text_area(WORDING["definition"], key="edit_definition", height=80)
        cols = st.columns(2)
        cols[0].text_area(
            WORDING["inclusion_criteria"], key="edit_inclusion_criteria", height=90
        )
        cols[1].text_area(
            WORDING["exclusion_criteria"], key="edit_exclusion_criteria", height=90
        )
        _undo_and_save(ctx, leaf, wording, edited=edited, wording_edited=wording_edited)

    shared.persist(kept_keys)
    return prompt_id, system, wording, edited


def _undo_and_save(
    ctx: Context, leaf: dict, wording: dict, *, edited: bool, wording_edited: bool
) -> None:
    """Undo the edits, or save the wording to the class."""
    cols = st.columns([1, 2, 3])
    if cols[0].button("Undo my edits", disabled=not edited):
        st.session_state.pop("judgement_context", None)
        st.rerun()
    if cols[1].button(
        "Save this wording to the class",
        disabled=not wording_edited,
        help="The class's saved definition and criteria become these; the next run "
        "or label mints a new version. The system text is part of the prompt, not "
        "the class, and is not saved.",
    ):
        try:
            ctx.api.update_concept(
                ctx.ontology["id"],
                leaf["id"],
                **{field: value or "" for field, value in wording.items()},
            )
            st.session_state.pop("judgement_context", None)
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)
