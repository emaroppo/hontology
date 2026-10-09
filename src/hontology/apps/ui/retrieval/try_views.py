"""Try a cutoff, the views: your own text, a corpus article, or one class's
articles, each ranked as the run ranks it, with what the cutoff keeps."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.retrieval.context import Context


def mark(value: bool | None) -> str:
    return {True: "✓", False: "✗", None: ""}[value]


def label_mark(value: bool | None) -> str:
    return {True: "✓ match", False: "✗ no", None: ""}[value]


# ---------------------------------------------------------------------------
# Try a cutoff: your text, a corpus article, or one class's articles
# ---------------------------------------------------------------------------


def show_article(ctx: Context, document_id: int, cutoff: dict) -> None:
    run = ctx.run
    assert run is not None
    try:
        pool = ctx.api.retrieval_document(run["id"], document_id, cutoff, ctx.annotator)
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


def show_own_text(ctx: Context, cutoff: dict) -> None:
    run = ctx.run
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
                cache[key] = ctx.api.retrieval_text(run["id"], body, cutoff)
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


def show_class(ctx: Context, leaf: dict, cutoff: dict) -> None:
    run = ctx.run
    assert run is not None
    if leaf.get("definition"):
        st.caption(leaf["definition"])
    try:
        rows = ctx.api.retrieval_concept(run["id"], leaf["id"], cutoff, ctx.annotator, limit=50)
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
