"""Evaluation: a retrieval version, scored live.

A version is the embedding settings, the leaves' wording and the ranking code; a
cutoff is a parameter, set by hand or loaded from a run that used the version.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.cutoff import (
    CUTOFF_FIELDS,
    EV_CUT_KEYS,
    cutoff_sliders,
    describe_cutoff,
    read_cutoff,
    seed_cutoff,
)
from hontology.apps.ui.retrieval.context import Context
from hontology.apps.ui.retrieval.evaluation_results import show_results


def show_evaluation(ctx: Context) -> None:
    api, annotator = ctx.api, ctx.annotator
    try:
        versions = api.retrieval_versions(ctx.ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not versions:
        st.info("No run of this ontology yet, so there is no retrieval version to score.")
        return

    by_version = {_describe(v): v for v in versions}
    version = by_version[st.selectbox("Retrieval version", list(by_version), key="ev_version")]
    st.caption(
        f"Ranking code {version['code']}. Runs that used it: "
        + ", ".join(str(r["id"]) for r in version["runs"])
        + "."
    )

    loaded, edited = _cutoff_panel(_presets(version))
    cutoff = {
        "selection": st.session_state["ev_selection"],
        "top_k": int(st.session_state["ev_top_k"]),
        "min_score": float(st.session_state["ev_min_score"]),
        "rel_margin": float(st.session_state["ev_rel_margin"]),
        "max_k": int(st.session_state["ev_max_k"]),
    }
    pool_size = int(st.session_state["ev_pool"])
    try:
        result = api.retrieval_evaluate(version["runs"][0]["id"], cutoff, pool_size, annotator)
        base = (
            api.retrieval_evaluate(version["runs"][0]["id"], loaded, pool_size, annotator)
            if edited
            else None
        )
    except ApiError as exc:
        st.error(exc.detail)
        return
    show_results(ctx, version, cutoff, result, base)


def _describe(v: dict) -> str:
    s_ = v["settings"]
    return (
        f"{v['version']} · {s_['embed_model']} · {s_['concept_fields']} · "
        f"{s_['embed_body_limit']:,} chars · {v['leaves']} leaves at "
        f"{v['ontology_version']} · {len(v['runs'])} run(s)"
    )


def _presets(version: dict) -> dict[str, dict]:
    """One preset per distinct cutoff among the version's runs."""
    presets: dict[str, dict] = {}
    for r in version["runs"]:
        c = r["cutoff"]
        detail = (
            f"top {c['top_k']}"
            if c["selection"] == "top-k"
            else f"min {c['min_score']}, margin {c['rel_margin']}, at most {c['max_k']}"
        )
        presets.setdefault(f"{c['selection']}: {detail}", c | {"runs": []})["runs"].append(
            r["id"]
        )
    return presets


def _load(presets: dict[str, dict], label: str) -> None:
    preset = presets[label]
    st.session_state["ev_loaded"] = {k: preset[k] for k in CUTOFF_FIELDS}
    seed_cutoff("ev_", preset)


def _cutoff_panel(presets: dict[str, dict]) -> tuple[dict, bool]:
    """The cutoff and pool depth, in a collapsible section, starting from a
    preset. Returns the preset loaded and whether the cutoff was edited from it."""
    # The cutoff and the preset it started from are widget values, dropped when the
    # tab is left: keep a copy, so an edit survives a visit to Try.
    ev_keys = ("ev_preset", *EV_CUT_KEYS, "ev_pool")
    shared.restore(ev_keys)
    if st.session_state.get("ev_preset") not in presets:
        st.session_state["ev_preset"] = next(iter(presets))

    loaded = st.session_state.get("ev_loaded")
    if (
        loaded is None
        or loaded["selection"] is None
        or any(key not in st.session_state for key in EV_CUT_KEYS)
    ):
        _load(presets, st.session_state["ev_preset"])
        loaded = st.session_state["ev_loaded"]
    st.session_state.setdefault("ev_pool", 20)

    current = read_cutoff("ev_")
    edited = current != loaded
    header = f"**Cutoff** · {describe_cutoff(current)} · " + (
        "edited" if edited else "the preset's own"
    )
    with st.expander(header, key="ev_cutoff_panel"):
        cols = st.columns([4, 1])
        preset_label = cols[0].selectbox(
            "Preset from a run",
            list(presets),
            format_func=lambda k: f"{k}  (runs {', '.join(map(str, presets[k]['runs']))})",
            key="ev_preset",
        )
        cols[1].markdown("&nbsp;")
        if cols[1].button("Load preset", help="Set the cutoff to this preset's."):
            _load(presets, preset_label)
            st.rerun()
        cols = st.columns(6)
        cutoff_sliders(cols, "ev_")
        cols[5].number_input(
            "Pool depth",
            1,
            100,
            key="ev_pool",
            help="How far down each article's ranking to look.",
        )
        st.caption(
            "Changes are compared with the loaded preset: recall and cost below show by "
            "how much they move."
        )
    shared.persist(ev_keys)
    return loaded, edited
