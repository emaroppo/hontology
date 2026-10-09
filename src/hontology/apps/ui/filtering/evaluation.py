"""Evaluation: a version of the links, link by link, against the labels, a
calendar and the whole corpus."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import panel
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.filtering.context import Context


def show_evaluation(ctx: Context) -> None:
    api, ontology_id = ctx.api, ctx.ontology_id
    try:
        listing = api.filtering_versions(ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    version = _pick_version(ctx, listing)

    annotator = panel.current(api).annotator
    payload = {"ontology_id": ontology_id, "version": version, "annotator": annotator}
    try:
        labels = api.filtering_evaluate("labels", **payload)
    except ApiError as exc:
        st.error(exc.detail)
        return
    _against_labels(labels)
    calendar_result, cost = _calendar_and_cost(ctx, version, payload)
    if calendar_result:
        _calendar_table(calendar_result)
    _link_table(labels, calendar_result, cost)


def _pick_version(ctx: Context, listing: dict) -> str | None:
    """The links to score: live (None), or a kept version; live links not kept
    yet can be kept from here."""
    live = listing["live"]
    options: dict[str, str | None] = {
        f"Live links ({live['n_links']})"
        + (f", the same as {live['is']}" if live["is"] else ""): None
    }
    for snap in listing["snapshots"]:
        used = f", used by runs {', '.join(map(str, snap['runs']))}" if snap["runs"] else ""
        options[
            f"{snap['version']} ({snap['n_links']} links, {snap['created_at'][:10]}{used})"
        ] = snap["version"]
    cols = st.columns([3, 1])
    version = options[cols[0].selectbox("Links", list(options), key="fe_version")]
    unkept = version is None and not live["is"] and live["n_links"]
    if unkept and cols[1].button(
        "Keep as a version", help="Snapshot the live links to compare later."
    ):
        try:
            kept = ctx.api.filtering_snapshot(ctx.ontology_id)
            st.session_state["filtering_message"] = f"Kept the live links as {kept['version']}."
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)
    return version


def _against_labels(labels: dict) -> None:
    st.markdown("**Against the labels**")
    cols = st.columns(3)
    cols[0].metric(
        "True matches admitted",
        f"{labels['positives_admitted']} of {labels['positives']}",
    )
    cols[1].metric(
        "Labelled articles admitted", f"{labels['documents_admitted']} of {labels['documents']}"
    )
    st.caption(
        "Labelled articles were mostly downloaded because the filter admitted them, so "
        "these counts flatter it, its misses most of all. The calendar below does not "
        "have that bias."
    )


def _calendar_and_cost(
    ctx: Context, version: str | None, payload: dict
) -> tuple[dict | None, dict | None]:
    """Score on a calendar and count the corpus cost, each on request and kept for
    the session; returns whichever have been computed."""
    key = (ctx.ontology_id, version)
    cache = st.session_state.setdefault("filter_eval", {})
    st.markdown("**Against the calendar and over the whole corpus** (a few minutes each)")
    cols = st.columns([2, 1, 1])
    calendar_path = cols[0].text_input(
        "Calendar", value="data/calendars/supply-chain-v2-small.csv", key="fe_calendar"
    )
    if cols[1].button("Score on the calendar"):
        try:
            with st.spinner("Walking every calendar window…"):
                cache[(key, "calendar")] = ctx.api.filtering_evaluate(
                    "calendar", **payload, calendar_path=calendar_path
                )
        except ApiError as exc:
            st.error(exc.detail)
    if cols[2].button("Count the cost"):
        try:
            with st.spinner("Walking every feed record…"):
                cache[(key, "cost")] = ctx.api.filtering_evaluate("cost", **payload)
        except ApiError as exc:
            st.error(exc.detail)
    return cache.get((key, "calendar")), cache.get((key, "cost"))


def _calendar_table(calendar_result: dict) -> None:
    st.dataframe(
        [
            {
                "Kind": kind,
                "Events": c["events"],
                "With articles in the window": c["observable"],
                "Reached by any link": c["reached"],
                "Reached by the event's own class": c["reached_by_own_class"],
            }
            for kind, c in calendar_result["by_kind"].items()
        ],
        hide_index=True,
    )
    missed = calendar_result["not_reached_by_own_class"]
    if missed:
        with st.expander(f"Events the own class's links miss ({len(missed)})"):
            for event in missed:
                note = "" if event["known_class"] else " (no class of that name)"
                st.markdown(
                    f"- {event['kind']} · {event['class']}{note}: {event['description']}"
                )


def _link_table(labels: dict, calendar_result: dict | None, cost: dict | None) -> None:
    st.markdown("**Link by link**")
    st.caption(
        "Against the labels for the class each link reaches: TP admitted true matches, "
        "FP admitted non-matches, FN true matches it does not admit, TN non-matches it "
        "rightly keeps out; unique TP, true matches no other link admits. Corpus columns "
        "count feed articles per code: admitted, and admitted by no other link, which "
        "removing the link would stop downloading."
    )
    rows = []
    for link in labels["links"]:
        code_key = f"{link['system']}:{link['code']}"
        row = {
            "Class": link["class"],
            "System": link["system"],
            "Code": link["code"],
            "TP": link["tp"],
            "FP": link["fp"],
            "FN": link["fn"],
            "TN": link["tn"],
            "Unique TP": link["unique_tp"],
        }
        if calendar_result:
            row["Calendar events"] = len(calendar_result["per_code"].get(code_key, []))
        if cost:
            counts = cost["codes"].get(code_key, {})
            row["Corpus admitted"] = counts.get("admitted", 0)
            row["Only this code"] = counts.get("only_this", 0)
            row["Downloaded"] = counts.get("downloaded", 0)
        rows.append(row)
    rows.sort(key=lambda r: (-r.get("Only this code", 0), -r["FP"]))
    st.dataframe(rows, hide_index=True)
