"""Settings that carry from page to page, in a panel at the top right.

The sidebar is navigation only. What more than one page reads, the ontology,
the run, the truth scores are computed against and the labelled sample, sits in
a popover that opens and closes from the top right corner of every page, its
button showing the current values. Everything a single page needs is in that
page's body, next to what it controls.

The entry script draws the panel once on every page view (`settings`), so the
values carry across pages as they are; a page reads them with `current`. A page
run on its own, as in a test, gets the defaults instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from hontology.apps.ui.client import Api, ApiError

DEFAULT_SAMPLE = "data/samples/sample-run566.json"
_VALUES = "shared:values"
# A page can ask for a value to change; the panel applies it before drawing,
# since a widget's value cannot be set once it is drawn in a run.
_NEXT = "shared:next:"


@dataclass(frozen=True)
class Params:
    ontology_id: int | None
    run_id: int | None
    # None for human labels, else a machine annotation set's name.
    annotator: str | None
    sample: str


def _choose(key: str, options: list) -> None:
    """Drop a value no longer offered (a run of the ontology just left) and
    apply a value a page asked for."""
    if _NEXT + key in st.session_state:
        wanted = st.session_state.pop(_NEXT + key)
        if wanted in options:
            st.session_state[key] = wanted
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def request(name: str, value: object) -> None:
    """Ask the panel to show *value* for *name* ("ontology", "run") on the next
    run; the caller then reruns."""
    st.session_state[_NEXT + f"settings_{name}"] = value


def _summary(ontologies: list[dict], runs: list[dict]) -> str:
    """The panel button's label: the values in force, at a glance. Drawn before
    the widgets, so a value not chosen yet is the one its widget will default to."""
    by_id = {o["id"]: o for o in ontologies}
    ontology = by_id.get(st.session_state.get("settings_ontology"), ontologies[0])
    parts = [ontology["name"]]
    own = [r["id"] for r in runs if r["ontology_id"] == ontology["id"]]
    run_id = st.session_state.get("settings_run")
    run_id = run_id if run_id in own else (own[0] if own else None)
    if run_id is not None:
        parts.append(f"run {run_id}")
    annotator = st.session_state.get("settings_truth")
    parts.append(f"machine: {annotator}" if annotator else "human labels")
    return " · ".join(parts)


def settings(api: Api) -> None:
    """Draw the shared settings panel. Called by the entry script on every page."""
    if not api.healthy():
        return
    ontologies = api.list_ontologies()
    if not ontologies:
        st.session_state[_VALUES] = Params(None, None, None, DEFAULT_SAMPLE)
        return
    all_runs = api.list_runs()
    _, corner = st.columns([3, 2])
    # A fixed key: the label changes with every choice, and without one each
    # change would make it a new panel and rebuild the widgets inside it.
    with (
        corner,
        st.popover(
            _summary(ontologies, all_runs),
            icon=":material/tune:",
            width="stretch",
            key="settings_panel",
        ),
    ):
        by_id = {o["id"]: o for o in ontologies}
        _choose("settings_ontology", list(by_id))
        ontology_id = st.selectbox(
            "Ontology",
            list(by_id),
            key="settings_ontology",
            format_func=lambda i: f"{by_id[i]['name']} ({by_id[i]['slug']})",
        )

        runs = {r["id"]: r for r in all_runs if r["ontology_id"] == ontology_id}
        run_id = None
        if runs:
            _choose("settings_run", list(runs))
            run_id = st.selectbox(
                "Run",
                list(runs),
                key="settings_run",
                format_func=lambda i: f"{i} · {runs[i]['name']} ({runs[i]['status']})",
                help="The run Retrieval explores, Judgement asks as, and Home shows alone.",
            )
        else:
            st.caption("No run of this ontology yet.")

        truths = api.truths(ontology_id)
        human = truths["human"]
        labels: dict[str | None, str] = {None: f"Human labels ({human['articles']} articles)"}
        descriptions = {}
        for found in truths["machine"]:
            labels[found["name"]] = f"Machine: {found['name']} ({found['articles']} articles)"
            descriptions[found["name"]] = found["description"]
        _choose("settings_truth", list(labels))
        annotator = st.radio(
            "Truth",
            list(labels),
            key="settings_truth",
            format_func=lambda value: labels[value],
            help="Human labels are the label bank, trusted and current only. A machine "
            "annotation set is an annotator's own labels: a score against it measures "
            "agreement with that annotator, not correctness.",
        )
        if annotator is not None and descriptions.get(annotator):
            st.caption(descriptions[annotator])

        # A blank path is never meant (it reads as the current folder), so it falls
        # back to the default; it can arrive blank from a session older than the
        # panel's fixed key, whose rebuilt fields could send an empty value.
        if not str(st.session_state.get("settings_sample") or "").strip():
            st.session_state["settings_sample"] = DEFAULT_SAMPLE
        sample = st.text_input(
            "Sample manifest",
            key="settings_sample",
            help="The labelled sample in its frozen order: what Home scores runs on and "
            "what Labelling works through.",
        )
        if not Path(sample).is_file():
            st.warning(f"No file at {sample}: Home and Labelling cannot read the sample.")
    st.session_state[_VALUES] = Params(ontology_id, run_id, annotator, sample)


def current(api: Api) -> Params:
    """The shared settings, or the defaults when no panel was drawn."""
    values = st.session_state.get(_VALUES)
    if values is not None:
        return values
    ontologies = api.list_ontologies()
    ontology_id = ontologies[0]["id"] if ontologies else None
    runs = [r for r in api.list_runs() if r["ontology_id"] == ontology_id]
    return Params(ontology_id, runs[0]["id"] if runs else None, None, DEFAULT_SAMPLE)


def ontology(api: Api, ontologies: list[dict]) -> dict:
    """The chosen ontology, as the listing gives it."""
    wanted = current(api).ontology_id
    return next((o for o in ontologies if o["id"] == wanted), ontologies[0])


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
