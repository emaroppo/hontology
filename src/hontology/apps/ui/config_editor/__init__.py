"""A run config, edited as a form or as JSON, the same config either way.

The page holds one config. The form writes it field by field; the JSON editor
writes it whole once the text parses and the API accepts it. Switching views
carries the config across, so an edit made in one shows in the other, and JSON
that does not parse is reported rather than lost or half applied.

Every choice the form offers comes from the API (`GET /runs/options`), so a new
prompt template or aggregation rule shows up here without touching this package.
"""

from __future__ import annotations

import json

import streamlit as st

from hontology.apps.ui.client import Api, ApiError
from hontology.apps.ui.config_editor.form import draw_form
from hontology.apps.ui.config_editor.state import (
    CONFIG,
    SHOWN,
    from_form,
    key,
    load,
    seed_form,
)

FORM, JSON = "Form", "JSON"


def editor(api: Api, ontology_id: int, versions: list[str]) -> tuple[dict, str | None]:
    """Draw the editor; returns the config and why it cannot run, if it cannot."""
    options = api.run_options()
    if CONFIG not in st.session_state:
        load(options["defaults"] | {"name": "baseline", "description": ""})
    # The view's one source is session state (a default too would make two), so
    # the last view chosen comes back when the page is returned to.
    if "cfg:view" not in st.session_state:
        st.session_state["cfg:view"] = st.session_state.get("keep:cfg:view", FORM)
    view = (
        st.segmented_control(
            "Edit as",
            [FORM, JSON],
            required=True,
            key="cfg:view",
            label_visibility="collapsed",
        )
        or FORM
    )
    st.session_state["keep:cfg:view"] = view
    # The fields are widget values, which Streamlit drops when the page is left;
    # the config itself is kept. So a view is redrawn from the config on entering
    # it, and also whenever its fields have been dropped.
    dropped = (
        key("json") not in st.session_state
        if view == JSON
        else any(key(path) not in st.session_state for path in ("name", "judge.model"))
    )
    if st.session_state.get(SHOWN) != view or dropped:
        # Entering a view: draw it from the config as it stands.
        if view == FORM:
            seed_form(st.session_state[CONFIG])
        else:
            st.session_state[key("json")] = json.dumps(st.session_state[CONFIG], indent=2)
        st.session_state[SHOWN] = view

    if view == FORM:
        draw_form(options, versions)
        st.session_state[CONFIG] = from_form()
        problem = None
    else:
        text = st.text_area("Run config (JSON)", key=key("json"), height=420)
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("the config must be a JSON object")
            api.preview_keys(parsed)  # the API's own validation
            st.session_state[CONFIG] = parsed
            problem = None
        except (ValueError, ApiError) as exc:
            detail = exc.detail if isinstance(exc, ApiError) else str(exc)
            problem = f"Not applied: {detail}. Fix it, or switch to the form to drop the edit."
            st.error(problem)
    return st.session_state[CONFIG], problem
