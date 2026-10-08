"""Retrieval: what a run's search ranked, and where its cutoff should fall.

Retrieval ranks an ontology's leaves against each article by embedding
similarity, and a cutoff decides which pairs go on to the judge. Every run stores
its whole ranked pool, so this page can move the cutoff after the fact without
embedding anything: the same ranking, a different line through it.

**Tune** compares a cutoff with the one the run was built with: how many pairs
it sends to the judge, and how many labelled true matches it keeps, split into
those the pool never ranked (no cutoff can recover them) and those the cutoff
dropped. **Explore** shows the ranking itself, one article or one class at a
time, under both cutoffs.

**Evaluation** scores a retrieval *version* rather than a run: the embedding
settings, the leaves' wording and the ranking code. A cutoff is a parameter, set
by hand or loaded from a run that used the version, and every labelled article
is ranked afresh from cached embeddings, so the score covers all labelled
articles, not only those one run happened to retrieve for.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import charts, shared
from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Retrieval", page_icon="🔎", layout="wide")

api = Api()

st.title("🔎 Retrieval")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()


# ---------------------------------------------------------------------------
# The ontology, run and truth chosen in Settings; the cutoff, in the page
# ---------------------------------------------------------------------------

ontology = shared.ontology(api, ontologies)
params = shared.current(api)
annotator = params.annotator
runs = api.retrieval_runs(ontology["id"])
run = next((r for r in runs if r["id"] == params.run_id), None)

CUT_KEYS = ("cut_selection", "cut_top_k", "cut_min_score", "cut_rel_margin", "cut_max_k")


def cutoff_controls() -> dict:
    """The cutoff Tune and Explore apply, starting from the run's own."""
    assert run is not None
    own = run["cutoff"]
    # The controls are drawn on two tabs and neither while Evaluation is open,
    # and a keyed widget left undrawn loses its value: keep a copy.
    for key in CUT_KEYS:
        if key not in st.session_state and f"keep:{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"keep:{key}"]
    if st.session_state.get("retrieval_cutoff_for") != run["id"]:
        st.session_state["retrieval_cutoff_for"] = run["id"]
        st.session_state["cut_selection"] = own["selection"]
        st.session_state["cut_top_k"] = own["top_k"]
        st.session_state["cut_min_score"] = float(own["min_score"])
        st.session_state["cut_rel_margin"] = float(own["rel_margin"])
        st.session_state["cut_max_k"] = own["max_k"]
    with st.container(border=True):
        cols = st.columns([2, 2, 2, 2, 2, 1])
        selection = cols[0].radio(
            "Cutoff",
            ["adaptive", "top-k"],
            key="cut_selection",
            help="Adaptive keeps the classes within a margin of the article's best score; "
            "top-k keeps a fixed number per article.",
        )
        # Every control is always drawn, disabled when it does not apply.
        adaptive = selection == "adaptive"
        cols[1].slider("Classes per article", 1, 20, key="cut_top_k", disabled=adaptive)
        cols[2].slider(
            "Minimum score",
            0.0,
            1.0,
            step=0.01,
            key="cut_min_score",
            disabled=not adaptive,
            help="Nothing below this is kept, however close to the best.",
        )
        cols[3].slider(
            "Margin below the best",
            0.0,
            0.5,
            step=0.01,
            key="cut_rel_margin",
            disabled=not adaptive,
            help="Keep classes scoring within this much of the article's best.",
        )
        cols[4].slider(
            "At most", 1, 20, key="cut_max_k", disabled=not adaptive, help="Per article."
        )
        if cols[5].button("Reset", help="Back to the run's own cutoff."):
            st.session_state.pop("retrieval_cutoff_for", None)
            for key in CUT_KEYS:
                st.session_state.pop(f"keep:{key}", None)
            st.rerun()
        st.caption(
            f"Run {run['id']}'s own: {own['selection']}"
            + (
                f", top {own['top_k']}"
                if own["selection"] == "top-k"
                else f", min {own['min_score']}, margin {own['rel_margin']}, "
                f"at most {own['max_k']}"
            )
            + ". Its pool holds the top 20 per article; a cutoff can only draw from that."
        )
    for key in CUT_KEYS:
        st.session_state[f"keep:{key}"] = st.session_state[key]
    return {
        "selection": st.session_state["cut_selection"],
        "top_k": st.session_state["cut_top_k"],
        "min_score": st.session_state["cut_min_score"],
        "rel_margin": st.session_state["cut_rel_margin"],
        "max_k": st.session_state["cut_max_k"],
    }


def no_run() -> bool:
    if run is None:
        st.info(
            "Pick a run that has retrieved something in Settings (top right): Tune and "
            "Explore "
            "work on a run's stored ranking."
        )
        return True
    return False


def mark(value: bool | None) -> str:
    return {True: "✓", False: "✗", None: ""}[value]


def label_mark(value: bool | None) -> str:
    return {True: "✓ match", False: "✗ no", None: ""}[value]


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------


def show_tune() -> None:
    if no_run():
        return
    assert run is not None
    cutoff = cutoff_controls()
    changed = cutoff != run["cutoff"]
    try:
        result = api.retrieval_report(run["id"], cutoff, annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    mine, theirs = result["setting"], result["run"]

    st.markdown("**Cost: pairs sent to the judge**")
    cols = st.columns(3)
    cols[0].metric(
        "Pairs kept",
        f"{mine['pairs_kept']:,}",
        delta=f"{mine['pairs_kept'] - theirs['pairs_kept']:+,}" if changed else None,
        delta_color="inverse",
    )
    cols[1].metric(
        "Per article",
        f"{mine['pairs_per_document']:.2f}",
        delta=f"{mine['pairs_per_document'] - theirs['pairs_per_document']:+.2f}"
        if changed
        else None,
        delta_color="inverse",
    )
    cols[2].metric(
        "Articles with none kept",
        f"{mine['documents_with_none']:,}",
        help="Never judged at all under this cutoff.",
    )

    labels, run_labels = mine["labels"], theirs["labels"]
    st.markdown("**Recall: labelled true matches that reach the judge**")
    if not labels["positives"]:
        st.info(
            "No labelled true matches on this run's articles, so recall cannot be "
            "measured. Try a machine annotation set as truth (Settings, top right)."
        )
    else:
        cols = st.columns(4)
        positives = labels["positives"]
        cols[0].metric(
            "Kept",
            f"{labels['positives_kept']} of {positives}",
            delta=(labels["positives_kept"] - run_labels["positives_kept"])
            if changed
            else None,
        )
        cols[1].metric("Recall at the cutoff", f"{labels['positives_kept'] / positives:.0%}")
        cols[2].metric(
            "Lost to the cutoff",
            labels["positives_in_pool"] - labels["positives_kept"],
            help="Ranked in the pool, but below the line. A looser cutoff recovers these.",
        )
        cols[3].metric(
            "Never ranked",
            positives - labels["positives_in_pool"],
            help="Not in the article's top 20 at all. No cutoff recovers these; only a "
            "better ranking (model, class wording, article text) does.",
        )
        kept_labelled = labels["kept_labelled"]
        st.caption(
            f"Over {labels['documents']} labelled article(s). Of the {kept_labelled} kept "
            f"pairs on them, {labels['kept_positive']} are true matches"
            + (f" ({labels['kept_positive'] / kept_labelled:.0%})." if kept_labelled else ".")
            + " A small labelled set moves in whole steps: read changes of one or two "
            "as noise."
        )

    st.markdown("**Classes kept per article**")
    histogram = {int(k): v for k, v in mine["kept_per_document"].items()}
    st.bar_chart(
        {"articles": [histogram.get(k, 0) for k in range(max(histogram, default=0) + 1)]},
        x_label="classes kept",
        y_label="articles",
    )


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------


def show_article(document_id: int, cutoff: dict) -> None:
    assert run is not None
    try:
        pool = api.retrieval_document(run["id"], document_id, cutoff, annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.markdown(f"**[{pool['title'] or pool['url']}]({pool['url']})**")
    if pool["missed"]:
        st.warning("Labelled true matches the pool never ranked: " + ", ".join(pool["missed"]))
    st.dataframe(
        [
            {
                "Rank": row["rank"],
                "Class": row["name"],
                "Score": row["score"],
                "Run kept": mark(row["selected"]),
                "This cutoff": mark(row["kept"]),
                "Label": label_mark(row["label"]),
            }
            for row in pool["pool"]
        ],
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.3f")},
    )
    with st.expander("Article text (the first 4,000 characters)"):
        st.text(pool["body"] or "No text stored for this article.")


def show_explore() -> None:
    if no_run():
        return
    assert run is not None
    cutoff = cutoff_controls()
    mode = st.radio("Look at", ["One article", "One class"], horizontal=True)
    if mode == "One article":
        try:
            labelled = api.retrieval_labelled(run["id"], annotator)
        except ApiError as exc:
            st.error(exc.detail)
            return
        cols = st.columns([3, 1])
        options = {
            f"{d['title'] or d['url'][:90]}"
            + (f"  ({d['positives']} match)" if d["positives"] == 1 else "")
            + (f"  ({d['positives']} matches)" if d["positives"] > 1 else ""): d["document_id"]
            for d in labelled
        }
        picked = (
            cols[0].selectbox(
                f"Labelled article ({len(labelled)})", list(options), key="retrieval_article"
            )
            if options
            else None
        )
        typed = cols[1].number_input("or article id", min_value=0, value=0, step=1)
        document_id = int(typed) if typed else (options[picked] if picked else None)
        if document_id is None:
            st.info("No labelled article in this run. Enter an article id.")
            return
        show_article(document_id, cutoff)
        return

    try:
        leaves = api.leaves(ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return
    by_name = {
        f"{leaf['families'][0]} › {leaf['name']}"
        if leaf["families"][0] != leaf["name"]
        else leaf["name"]: leaf
        for leaf in leaves
    }
    leaf = by_name[st.selectbox("Class", list(by_name), key="retrieval_class")]
    if leaf.get("definition"):
        st.caption(leaf["definition"])
    try:
        rows = api.retrieval_concept(run["id"], leaf["id"], cutoff, annotator, limit=50)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.caption(
        "The 50 articles that score this class highest. Rank is the class's place "
        "among the article's own classes, which is what the cutoff looks at."
    )
    st.dataframe(
        [
            {
                "Score": row["score"],
                "Rank in article": row["rank"],
                "Run kept": mark(row["selected"]),
                "This cutoff": mark(row["kept"]),
                "Label": label_mark(row["label"]),
                "Article": row["title"] or row["url"],
                "Link": row["url"],
                "Id": row["document_id"],
            }
            for row in rows
        ],
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.3f"),
            "Link": st.column_config.LinkColumn(display_text="open"),
        },
    )


# ---------------------------------------------------------------------------
# Evaluation: a retrieval version, scored live
# ---------------------------------------------------------------------------


def show_evaluation() -> None:
    try:
        versions = api.retrieval_versions(ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not versions:
        st.info("No run of this ontology yet, so there is no retrieval version to score.")
        return

    def describe(v: dict) -> str:
        s_ = v["settings"]
        return (
            f"{v['version']} · {s_['embed_model']} · {s_['concept_fields']} · "
            f"{s_['embed_body_limit']:,} chars · {v['leaves']} leaves at "
            f"{v['ontology_version']} · {len(v['runs'])} run(s)"
        )

    by_version = {describe(v): v for v in versions}
    version = by_version[st.selectbox("Retrieval version", list(by_version), key="ev_version")]
    st.caption(
        f"Ranking code {version['code']}. Runs that used it: "
        + ", ".join(str(r["id"]) for r in version["runs"])
        + "."
    )

    # Presets: one per distinct cutoff among the version's runs.
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
    cols = st.columns([3, 1])
    preset_label = cols[0].selectbox(
        "Preset from a run",
        list(presets),
        format_func=lambda k: f"{k}  (runs {', '.join(map(str, presets[k]['runs']))})",
        key="ev_preset",
    )
    if cols[1].button("Load preset") or "ev_selection" not in st.session_state:
        preset = presets[preset_label]
        st.session_state["ev_selection"] = preset["selection"]
        st.session_state["ev_top_k"] = preset["top_k"]
        st.session_state["ev_min_score"] = float(preset["min_score"])
        st.session_state["ev_rel_margin"] = float(preset["rel_margin"])
        st.session_state["ev_max_k"] = preset["max_k"]

    cols = st.columns(6)
    selection = cols[0].radio("Selection", ["adaptive", "top-k"], key="ev_selection")
    adaptive = selection == "adaptive"
    cols[1].number_input("Top k", 1, 50, key="ev_top_k", disabled=adaptive)
    cols[2].number_input(
        "Min score", 0.0, 1.0, step=0.01, key="ev_min_score", disabled=not adaptive
    )
    cols[3].number_input(
        "Margin", 0.0, 1.0, step=0.01, key="ev_rel_margin", disabled=not adaptive
    )
    cols[4].number_input("At most", 1, 50, key="ev_max_k", disabled=not adaptive)
    pool_size = cols[5].number_input(
        "Pool depth", 1, 100, value=20, help="How far down each article's ranking to look."
    )
    cutoff = {
        "selection": selection,
        "top_k": int(st.session_state["ev_top_k"]),
        "min_score": float(st.session_state["ev_min_score"]),
        "rel_margin": float(st.session_state["ev_rel_margin"]),
        "max_k": int(st.session_state["ev_max_k"]),
    }
    try:
        result = api.retrieval_evaluate(
            version["runs"][0]["id"], cutoff, int(pool_size), annotator
        )
    except ApiError as exc:
        st.error(exc.detail)
        return

    docs = result["documents"]
    if not result["positives"]:
        st.info(
            f"No labelled true match among the {docs['labelled']} labelled article(s) on "
            "this version's leaves. Try a machine annotation set as truth (Settings, "
            "top right)."
        )
        return

    def interval(block: dict) -> str:
        low, high = block["ci"]
        return f"{block['rate']:.0%} ({low:.0%}–{high:.0%})" if low is not None else "—"

    cols = st.columns(5)
    cols[0].metric(
        "Recall at the cutoff",
        interval(result["recall"]),
        help="Labelled true matches kept, with a 95% Wilson interval.",
    )
    cols[1].metric("Recall in the pool", interval(result["pool_recall"]))
    cols[2].metric("Lost to the cutoff", result["lost_to_cutoff"])
    cols[3].metric("Never ranked", result["never_ranked"])
    cols[4].metric(
        "Pairs per article",
        f"{result['pairs_per_document']:.2f}" if result["pairs_per_document"] else "—",
        help="What the cutoff sends to the judge, on average.",
    )
    st.caption(
        f"{result['recall']['found']} of {result['positives']} labelled true matches kept, "
        f"over {docs['ranked']} labelled article(s)"
        + (
            f"; {docs['not_embedded']} not embedded under this version, so left out"
            if docs["not_embedded"]
            else ""
        )
        + f". Of the {result['kept_labelled']} kept labelled pairs, "
        f"{result['kept_positive']} are true matches."
    )
    st.markdown("**Recall at each depth of the ranking**")
    st.caption(
        "Labelled true matches within the top k classes of their article: the ranking's "
        "quality before any cutoff. Where it flattens, a deeper cutoff stops paying."
    )
    st.altair_chart(charts.recall_at_k(result["recall_at_k"]), use_container_width=True)
    st.markdown("**By class**")
    st.dataframe(
        [
            {
                "Class": row["class"],
                "True matches": row["positives"],
                "In the pool": row["in_pool"],
                "Kept": row["kept"],
            }
            for row in result["per_class"]
        ],
        hide_index=True,
    )


evaluation_tab, tune_tab, explore_tab = st.tabs(
    ["Evaluation", "Tune a run's cutoff", "Explore the ranking"],
    key="retrieval_tab",
    on_change="rerun",
)
if evaluation_tab.open:
    with evaluation_tab:
        show_evaluation()
if tune_tab.open:
    with tune_tab:
        show_tune()
if explore_tab.open:
    with explore_tab:
        show_explore()
