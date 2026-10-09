"""A retrieval cutoff: its widgets, their keys and a one-line description.

Retrieval draws the same five controls twice, for Try (keys ``cut_*``) and for
Evaluation (keys ``ev_*``); each set of keys is its prefix and a field name.
"""

from __future__ import annotations

import streamlit as st

CUTOFF_FIELDS = ("selection", "top_k", "min_score", "rel_margin", "max_k")
CUT_KEYS = tuple(f"cut_{field}" for field in CUTOFF_FIELDS)
EV_CUT_KEYS = tuple(f"ev_{field}" for field in CUTOFF_FIELDS)

_HELP = {
    "selection": "Adaptive keeps the classes within a margin of the article's best score; "
    "top-k keeps a fixed number per article.",
    "min_score": "Nothing below this is kept, however close to the best.",
    "rel_margin": "Keep classes scoring within this much of the article's best.",
    "max_k": "Per article.",
}


def describe_cutoff(cutoff: dict) -> str:
    if cutoff["selection"] == "top-k":
        return f"top {cutoff['top_k']}"
    return (
        f"adaptive, min {cutoff['min_score']:g}, margin {cutoff['rel_margin']:g}, "
        f"at most {cutoff['max_k']}"
    )


def seed_cutoff(prefix: str, cutoff: dict) -> None:
    """Set the controls under *prefix* to *cutoff*."""
    st.session_state[f"{prefix}selection"] = cutoff["selection"]
    st.session_state[f"{prefix}top_k"] = cutoff["top_k"]
    st.session_state[f"{prefix}min_score"] = float(cutoff["min_score"])
    st.session_state[f"{prefix}rel_margin"] = float(cutoff["rel_margin"])
    st.session_state[f"{prefix}max_k"] = cutoff["max_k"]


def read_cutoff(prefix: str) -> dict:
    """The cutoff the controls under *prefix* hold, as they hold it."""
    return {field: st.session_state[f"{prefix}{field}"] for field in CUTOFF_FIELDS}


def cutoff_sliders(cols, prefix: str, *, helps: bool = False) -> None:
    """The five controls, in ``cols[0]`` to ``cols[4]``. Every control is always
    drawn, disabled when it does not apply."""
    help_ = _HELP if helps else {}
    selection = cols[0].radio(
        "Selection",
        ["adaptive", "top-k"],
        key=f"{prefix}selection",
        help=help_.get("selection"),
    )
    adaptive = selection == "adaptive"
    cols[1].slider("Classes per article", 1, 20, key=f"{prefix}top_k", disabled=adaptive)
    cols[2].slider(
        "Minimum score",
        0.0,
        1.0,
        step=0.01,
        key=f"{prefix}min_score",
        disabled=not adaptive,
        help=help_.get("min_score"),
    )
    cols[3].slider(
        "Margin below the best",
        0.0,
        0.5,
        step=0.01,
        key=f"{prefix}rel_margin",
        disabled=not adaptive,
        help=help_.get("rel_margin"),
    )
    cols[4].slider(
        "At most", 1, 20, key=f"{prefix}max_k", disabled=not adaptive, help=help_.get("max_k")
    )
