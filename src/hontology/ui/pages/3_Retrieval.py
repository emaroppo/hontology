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
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

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
# Sidebar: ontology, run, truth, cutoff
# ---------------------------------------------------------------------------

with st.sidebar:
    by_label = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    ontology = by_label[st.selectbox("Ontology", list(by_label), key="retrieval_ontology")]
    runs = api.retrieval_runs(ontology["id"])
    if not runs:
        st.info("No run of this ontology has retrieved anything yet.")
        st.stop()
    by_run = {f"{r['id']} · {r['name']} · {r['documents']:,} articles": r for r in runs}
    run = by_run[st.selectbox("Run", list(by_run), key="retrieval_run")]

    st.divider()
    truth_source = st.radio(
        "Truth",
        ["Label bank", "Labels file"],
        horizontal=True,
        help="A whole-document labels file (as `eval arms --labels` takes) is used in "
        "place of the label bank, for a sample labelled elsewhere.",
    )
    labels_csv: str | None = None
    if truth_source == "Labels file":
        path = st.text_input("Labels file", value="data/labels/claude-blind-001-320.csv")
        try:
            labels_csv = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            st.error(f"Cannot read it: {exc}")
            st.stop()

    st.divider()
    st.subheader("Cutoff")
    own = run["cutoff"]
    # A different run starts from its own cutoff.
    if st.session_state.get("retrieval_cutoff_for") != run["id"] or st.button(
        "Reset to the run's own"
    ):
        st.session_state["retrieval_cutoff_for"] = run["id"]
        st.session_state["cut_selection"] = own["selection"]
        st.session_state["cut_top_k"] = own["top_k"]
        st.session_state["cut_min_score"] = float(own["min_score"])
        st.session_state["cut_rel_margin"] = float(own["rel_margin"])
        st.session_state["cut_max_k"] = own["max_k"]
    selection = st.radio(
        "Selection",
        ["adaptive", "top-k"],
        key="cut_selection",
        horizontal=True,
        help="Adaptive keeps the classes within a margin of the article's best score; "
        "top-k keeps a fixed number per article.",
    )
    # Every control is always drawn, disabled when it does not apply: a keyed
    # widget left undrawn loses its value on the next rerun.
    adaptive = selection == "adaptive"
    st.slider("Classes per article", 1, 20, key="cut_top_k", disabled=adaptive)
    st.slider(
        "Minimum score",
        0.0,
        1.0,
        step=0.01,
        key="cut_min_score",
        disabled=not adaptive,
        help="Nothing below this is kept, however close to the best.",
    )
    st.slider(
        "Margin below the best",
        0.0,
        0.5,
        step=0.01,
        key="cut_rel_margin",
        disabled=not adaptive,
        help="Keep classes scoring within this much of the article's best.",
    )
    st.slider(
        "At most",
        1,
        20,
        key="cut_max_k",
        disabled=not adaptive,
        help="Classes kept per article, at most.",
    )
    st.caption(
        f"The run's own: {own['selection']}"
        + (
            f", top {own['top_k']}"
            if own["selection"] == "top-k"
            else f", min {own['min_score']}, margin {own['rel_margin']}, at most {own['max_k']}"
        )
        + ". Its pool holds the top 20 per article; a cutoff can only draw from that."
    )

cutoff = {
    "selection": st.session_state["cut_selection"],
    "top_k": st.session_state["cut_top_k"],
    "min_score": st.session_state["cut_min_score"],
    "rel_margin": st.session_state["cut_rel_margin"],
    "max_k": st.session_state["cut_max_k"],
}
changed = cutoff != own


def mark(value: bool | None) -> str:
    return {True: "✓", False: "✗", None: ""}[value]


def label_mark(value: bool | None) -> str:
    return {True: "✓ match", False: "✗ no", None: ""}[value]


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------


def show_tune() -> None:
    try:
        result = api.retrieval_report(run["id"], cutoff, labels_csv)
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
            "measured. Try a labels file as truth."
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


def show_article(document_id: int) -> None:
    try:
        pool = api.retrieval_document(run["id"], document_id, cutoff, labels_csv)
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
    mode = st.radio("Look at", ["One article", "One class"], horizontal=True)
    if mode == "One article":
        try:
            labelled = api.retrieval_labelled(run["id"], labels_csv)
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
        show_article(document_id)
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
        rows = api.retrieval_concept(run["id"], leaf["id"], cutoff, labels_csv, limit=50)
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


tune_tab, explore_tab = st.tabs(
    ["Tune the cutoff", "Explore the ranking"], key="retrieval_tab", on_change="rerun"
)
if tune_tab.open:
    with tune_tab:
        show_tune()
if explore_tab.open:
    with explore_tab:
        show_explore()
