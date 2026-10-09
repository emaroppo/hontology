"""What every page draws the same way: its preamble, tabs, kept widget values,
and a few small helpers several pages share.

The settings that carry from page to page are in `panel`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import streamlit as st

from hontology.apps.ui.client import Api, ApiError


def page(
    title: str, icon: str, *, caption: str | None = None, need_ontology: bool = True
) -> tuple[Api, list[dict]]:
    """A page's preamble: its config and title, then a stop if the API is down,
    or, when *need_ontology*, if there is no ontology yet. Returns the client and
    the ontologies."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    api = Api()
    st.title(f"{icon} {title}")
    if caption:
        st.caption(caption)
    if not api.healthy():
        st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
        st.stop()
    ontologies = api.list_ontologies()
    if need_ontology and not ontologies:
        st.info("No ontologies yet. Create one on the **Ontology** page first.")
        st.stop()
    return api, ontologies


def act(fn: Callable, *args, success: str | None = None, **kwargs) -> None:
    """Call the API and rerun, or show why it refused."""
    try:
        fn(*args, **kwargs)
        if success:
            st.success(success)
        st.rerun()
    except ApiError as exc:
        st.error(exc.detail)


def lazy_tabs(key: str, tabs: dict[str, Callable[[], None]]) -> None:
    """Tabs of which only the open one is drawn at all."""
    for tab, draw in zip(
        st.tabs(list(tabs), key=key, on_change="rerun"), tabs.values(), strict=True
    ):
        if tab.open:
            with tab:
                draw()


# A keyed widget left undrawn (its tab or page not open) loses its value, so a
# page keeps a copy under "keep:<key>" and restores it before drawing.


def restore(keys: Iterable[str]) -> None:
    for key in keys:
        if key not in st.session_state and f"keep:{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"keep:{key}"]


def persist(keys: Iterable[str]) -> None:
    for key in keys:
        st.session_state[f"keep:{key}"] = st.session_state[key]


def ci_text(value: float | None, ci: list | None) -> str:
    if value is None:
        return "—"
    if ci and ci[0] is not None:
        return f"{value:.2f} ({ci[0]:.2f}–{ci[1]:.2f})"
    return f"{value:.2f}"


def tree_order(classes: list[dict]) -> list[tuple[int, dict]]:
    """``(depth, class)`` depth first from the top level, children by name.

    A class with several parents appears under each of them.
    """
    by_name = {c["name"]: c for c in classes}
    out: list[tuple[int, dict]] = []

    def visit(concept: dict, depth: int) -> None:
        out.append((depth, concept))
        for child in concept["children"]:
            visit(by_name[child], depth + 1)

    for concept in classes:
        if not concept["parents"]:
            visit(concept, 0)
    return out


def leaf_options(leaves: list[dict]) -> dict[str, dict]:
    """Leaves by a label naming their family, for a selectbox."""
    return {
        f"{leaf['families'][0]} › {leaf['name']}"
        if leaf["families"][0] != leaf["name"]
        else leaf["name"]: leaf
        for leaf in leaves
    }


def own_text(where_sent: str) -> tuple[str, str]:
    """A pasted article: its title and text, kept across pages for the session,
    so text pasted on Retrieval is there on Judgement. Returns (title, text)."""
    keys = ("own:title", "own:text")
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"keep:{key}", "")
    st.text_input(
        "Title",
        key="own:title",
        placeholder="Optional · sent to the judge, not used by retrieval",
        help="Sent to the judge on the prompt's title line, as a corpus article's title "
        "is. Retrieval ranks the text alone, as it does a corpus article's body, so "
        "the title changes nothing there. Leave it empty if the headline is already "
        "the first line of the text.",
    )
    st.text_area(
        "Article text",
        key="own:text",
        height=220,
        placeholder="Paste an article from anywhere.",
    )
    st.caption(f"Not stored anywhere. {where_sent}")
    persist(keys)
    return st.session_state["own:title"].strip(), st.session_state["own:text"]
