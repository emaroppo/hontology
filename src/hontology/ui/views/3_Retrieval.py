"""Retrieval: what a run's search ranks, and where its cutoff should fall.

Retrieval ranks an ontology's leaves against each article by embedding
similarity, and a cutoff decides which pairs go on to the judge. Laid out as
Judgement is:

**Try a cutoff** shows the ranking for one thing at a time, your own pasted
text first, or a corpus article, or the articles that rank one class highest,
as the run chosen in Settings ranks them. The cutoff sits in a collapsible
section, starting from that run's own. Corpus articles need no embedding: every
run stores its whole ranked pool, so a cutoff is a different line through a
ranking that already exists.

**Evaluation** scores a retrieval *version* rather than a run: the embedding
settings, the leaves' wording and the ranking code. A cutoff is a parameter, set
by hand or loaded from a run that used the version. Recall is computed live over
every labelled article; cost, the pairs sent to the judge, over the articles of
the run whose ranking the version is.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import charts, shared
from hontology.ui.client import ApiError
from hontology.ui.cutoff import (
    CUT_KEYS,
    CUTOFF_FIELDS,
    EV_CUT_KEYS,
    cutoff_sliders,
    describe_cutoff,
    read_cutoff,
    seed_cutoff,
)

api, ontologies = shared.page("Retrieval", "🔎")


# ---------------------------------------------------------------------------
# The ontology, run and truth chosen in Settings; the cutoff, in the page
# ---------------------------------------------------------------------------

ontology = shared.ontology(api, ontologies)
params = shared.current(api)
annotator = params.annotator
runs = api.retrieval_runs(ontology["id"])
run = next((r for r in runs if r["id"] == params.run_id), None)


def cutoff_controls() -> dict:
    """The cutoff Try applies, in a collapsible section, starting from the run's own."""
    assert run is not None
    own = run["cutoff"]
    # Kept across a visit to Evaluation, which leaves these widgets undrawn.
    shared.restore(CUT_KEYS)
    if st.session_state.get("retrieval_cutoff_for") != run["id"]:
        st.session_state["retrieval_cutoff_for"] = run["id"]
        seed_cutoff("cut_", own)
    current = read_cutoff("cut_")
    edited = current != {k: own[k] for k in current}
    state = "edited" if edited else f"run {run['id']}'s own"
    header = f"**Cutoff** · {describe_cutoff(current)} · {state}"
    with st.expander(header, key="retrieval_cutoff_panel"):
        cutoff_sliders(st.columns(5), "cut_", helps=True)
        cols = st.columns([1, 4])
        if cols[0].button("Back to the run's own", disabled=not edited):
            st.session_state.pop("retrieval_cutoff_for", None)
            for key in CUT_KEYS:
                st.session_state.pop(f"keep:{key}", None)
            st.rerun()
        cols[1].caption(
            f"Run {run['id']}'s own: {describe_cutoff(own)}. Its pool holds the top 20 "
            "classes per article; a cutoff can only draw from that."
        )
    shared.persist(CUT_KEYS)
    return read_cutoff("cut_")


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
# Try a cutoff: your text, a corpus article, or one class's articles
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


def show_own_text(cutoff: dict) -> None:
    assert run is not None
    _, body = shared.own_text(
        "The text is sent to this run's embedding model, ranked and dropped."
    )
    if not body.strip():
        st.info(
            "Paste an article to see how this run's retrieval ranks the classes against "
            "it, and which the cutoff would send to the judge."
        )
        return
    # Ranking costs an embedding call; keep each result for its text, run and cutoff.
    cache = st.session_state.setdefault("own:rankings", {})
    key = (run["id"], body, tuple(sorted(cutoff.items())))
    if key not in cache:
        try:
            with st.spinner("Embedding and ranking…"):
                cache[key] = api.retrieval_text(run["id"], body, cutoff)
        except ApiError as exc:
            st.error(exc.detail)
            return
    result = cache[key]
    kept = [row["name"] for row in result["ranking"] if row["kept"]]
    st.markdown(
        f"**{len(kept)} class(es) would go to the judge**: " + (", ".join(kept) or "none")
    )
    cut_note = (
        f", cut from {result['characters_given']:,}"
        if result["characters_given"] > result["characters_used"]
        else ""
    )
    st.caption(
        f"Ranked with retrieval version {result['version']} (run {result['source_run']}'s), "
        f"on the first {result['characters_used']:,} characters{cut_note}, as an "
        "article would be."
    )
    st.dataframe(
        [
            {
                "Rank": row["rank"],
                "Class": row["name"],
                "Score": row["score"],
                "To the judge": "✓" if row["kept"] else "",
            }
            for row in result["ranking"]
        ],
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.3f")},
    )
    st.page_link(
        "views/4_Judgement.py",
        label="Ask the judge about this text",
        icon="⚖️",
        help="The text carries over to Judgement's Your text.",
    )


def show_class(leaf: dict, cutoff: dict) -> None:
    assert run is not None
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


def show_try() -> None:
    if no_run():
        return
    assert run is not None
    cols = st.columns([3, 2])
    look = cols[1].radio(
        "Look at",
        ["Your text", "A corpus article", "A class's articles"],
        horizontal=True,
        key="retrieval_look",
        help="Your text: an article pasted in, ranked as a corpus article would be. "
        "It is shared with Judgement's Your text.",
    )
    document_id = None
    leaf = None
    if look == "A corpus article":
        try:
            labelled = api.retrieval_labelled(run["id"], annotator)
        except ApiError as exc:
            st.error(exc.detail)
            return
        options = {
            d["document_id"]: f"{d['title'] or d['url'][:90]}"
            + (f"  ({d['positives']} match)" if d["positives"] == 1 else "")
            + (f"  ({d['positives']} matches)" if d["positives"] > 1 else "")
            for d in labelled
        }
        if not options:
            cols[0].info("No labelled article in this run.")
            return
        document_id = cols[0].selectbox(
            f"Labelled article ({len(options)})",
            list(options),
            format_func=lambda i: options[i],
            key="retrieval_article",
        )
    elif look == "A class's articles":
        try:
            leaves = api.leaves(ontology["id"])
        except ApiError as exc:
            st.error(exc.detail)
            return
        by_name = shared.leaf_options(leaves)
        leaf = by_name[cols[0].selectbox("Class", list(by_name), key="retrieval_class")]
    else:
        cols[0].caption(
            f"Ranked as run {run['id']}'s retrieval ranks an article: its embedding "
            "model, class wording and article length."
        )

    cutoff = cutoff_controls()
    if look == "Your text":
        show_own_text(cutoff)
    elif document_id is not None:
        show_article(document_id, cutoff)
    elif leaf is not None:
        show_class(leaf, cutoff)


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
    # The cutoff and the preset it started from are widget values, dropped when the
    # tab is left: keep a copy, so an edit survives a visit to Try.
    ev_keys = ("ev_preset", *EV_CUT_KEYS, "ev_pool")
    shared.restore(ev_keys)
    if st.session_state.get("ev_preset") not in presets:
        st.session_state["ev_preset"] = next(iter(presets))

    def load(label: str) -> None:
        preset = presets[label]
        st.session_state["ev_loaded"] = {k: preset[k] for k in CUTOFF_FIELDS}
        seed_cutoff("ev_", preset)

    loaded = st.session_state.get("ev_loaded")
    if (
        loaded is None
        or loaded["selection"] is None
        or any(key not in st.session_state for key in EV_CUT_KEYS)
    ):
        load(st.session_state["ev_preset"])
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
            load(preset_label)
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

    def change(now: float, then: float, fmt: str = "+d") -> str | None:
        return format(now - then, fmt) if base is not None and now != then else None

    cols = st.columns(5)
    cols[0].metric(
        "Recall at the cutoff",
        interval(result["recall"]),
        delta=change(result["recall"]["found"], base["recall"]["found"]) if base else None,
        help="Labelled true matches kept, with a 95% Wilson interval. The arrow is the "
        "change in true matches kept, against the loaded preset.",
    )
    cols[1].metric("Recall in the pool", interval(result["pool_recall"]))
    cols[2].metric(
        "Lost to the cutoff",
        result["lost_to_cutoff"],
        delta=change(result["lost_to_cutoff"], base["lost_to_cutoff"]) if base else None,
        delta_color="inverse",
    )
    cols[3].metric("Never ranked", result["never_ranked"])
    cols[4].metric(
        "Pairs per article",
        f"{result['pairs_per_document']:.2f}" if result["pairs_per_document"] else "—",
        delta=change(result["pairs_per_document"], base["pairs_per_document"], "+.2f")
        if base and result["pairs_per_document"] and base["pairs_per_document"]
        else None,
        delta_color="inverse",
        help="What the cutoff sends to the judge, on average, on the labelled articles.",
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
    # Cost: what the cutoff sends to the judge, over every article the version's
    # ranking was built for (its source run's stored pool).
    source_id = version["source_run"]
    try:
        cost = api.retrieval_report(source_id, cutoff, annotator)
    except ApiError as exc:
        st.error(exc.detail)
    else:
        mine, theirs = cost["setting"], cost["run"]
        changed = cutoff != theirs["cutoff"]
        st.markdown(f"**Cost: pairs sent to the judge, on run {source_id}'s articles**")
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
        st.caption(
            f"Over its {mine['documents']:,} articles"
            + (
                f", against its own cutoff ({describe_cutoff(theirs['cutoff'])})."
                if changed
                else ", which used this cutoff."
            )
        )
        histogram = {int(k): v for k, v in mine["kept_per_document"].items()}
        with st.expander("Classes kept per article", key="retrieval_histogram"):
            st.bar_chart(
                {
                    "articles": [
                        histogram.get(k, 0) for k in range(max(histogram, default=0) + 1)
                    ]
                },
                x_label="classes kept",
                y_label="articles",
            )

    st.markdown("**Recall at each depth of the ranking**")
    st.caption(
        "Labelled true matches within the top k classes of their article: the ranking's "
        "quality before any cutoff. Where it flattens, a deeper cutoff stops paying."
    )
    st.altair_chart(charts.recall_at_k(result["recall_at_k"]), width="stretch")
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


shared.lazy_tabs("retrieval_tab", {"Try a cutoff": show_try, "Evaluation": show_evaluation})
