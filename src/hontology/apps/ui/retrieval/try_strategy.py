"""Try a retrieval strategy: routing or extraction on one article or your text.

The embedding cutoff is the first step; what the strategy's later steps do with
it is shown step by step, ending in what each deciding call would be shown.
Those steps ask a decoder, so the page runs them only on request. The
embedding model and the decoder are picked apart, starting from the run's own.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.model_picker import DECODER, EMBEDDING
from hontology.apps.ui.retrieval.context import Context

EMBEDDING_ONLY = "Embedding cutoff only"


def strategy_picker(ctx: Context, column) -> str | None:
    """The prompt whose retrieval to try; None for the embedding cutoff alone."""
    try:
        strategies = ctx.api.retrieval_strategies()
    except ApiError as exc:
        st.error(exc.detail)
        return None
    top_down = [s["prompt_id"] for s in strategies if s["mode"] in ("hierarchical", "extract")]
    picked = column.selectbox(
        "Strategy",
        [EMBEDDING_ONLY, *top_down],
        key="retrieval_strategy",
        help="A routing or extraction prompt adds its own steps after the embedding "
        "cutoff, and asks the decoder to take them.",
    )
    return None if picked == EMBEDDING_ONLY else picked


def show_strategy(
    ctx: Context,
    prompt_id: str,
    cutoff: dict,
    models: dict[str, tuple[str, str]],
    *,
    document_id: int | None = None,
) -> None:
    run = ctx.run
    assert run is not None
    embedding, decoder = models[EMBEDDING], models[DECODER]
    text = None
    if document_id is None:
        _, text = shared.own_text("The text is sent to this run's models and dropped.")
        if not text.strip():
            st.info("Paste an article to see what this strategy would show the judge.")
            return
    cache = st.session_state.setdefault("own:strategies", {})
    key = (
        run["id"],
        prompt_id,
        text,
        document_id,
        tuple(sorted(cutoff.items())),
        embedding,
        decoder,
    )
    if key not in cache:
        st.caption("Routing and extraction ask the decoder: this costs model calls.")
        if not st.button("Run retrieval", type="primary"):
            return
        try:
            with st.spinner("Retrieving…"):
                cache[key] = ctx.api.retrieval_strategy(
                    run["id"],
                    prompt_id,
                    cutoff,
                    text=text,
                    document_id=document_id,
                    embedding=embedding,
                    decoder=decoder,
                )
        except ApiError as exc:
            st.error(exc.detail)
            return
    _show_result(cache[key])


def _show_result(result: dict) -> None:
    spend = result["spend"]
    st.caption(
        f"{result['strategy']}: {' → '.join(result['steps'])} · {spend['calls']} model "
        f"call(s), {spend['input_tokens']:,} tokens in, {spend['output_tokens']:,} out, "
        f"{spend['seconds']:.1f} s · embedding {result['embed_provider']} "
        f"{result['embed_model']}, decoder {result['provider']} {result['model']}"
    )
    if result["error"]:
        st.error(f"Stopped: {result['error']}")
    elif result["stopped"]:
        st.warning(f"Stopped: {result['stopped']}. Nothing goes to the judge.")
    shown = result["shown"]
    st.markdown(
        f"**{len(shown)} class(es) would be shown to the judge**: "
        + (", ".join(shown) or "none")
    )
    if result["offers"]:
        st.dataframe(
            [
                {
                    "Deciding call": n,
                    "Event": offer["event"] or "",
                    "Shown": ", ".join(offer["classes"]),
                }
                for n, offer in enumerate(result["offers"], start=1)
            ],
            hide_index=True,
        )
    if result["routing"]:
        with st.expander(f"Routing calls ({len(result['routing'])})"):
            st.dataframe(
                [
                    {
                        "Asked": ", ".join(call["asked"]),
                        "Yes": ", ".join(call["yes"]),
                        "Error": call["error"] or "",
                    }
                    for call in result["routing"]
                ],
                hide_index=True,
            )
    if result["events"]:
        with st.expander(f"Events ({len(result['events'])})", expanded=True):
            st.dataframe(
                [
                    {
                        "Event": event["description"],
                        "Status": event["status"] or "",
                        "Routed through": ", ".join(event["routed_through"]),
                        "Offered": ", ".join(event["offered"]) or "nothing (rejected)",
                        "Error": event["error"] or "",
                    }
                    for event in result["events"]
                ],
                hide_index=True,
            )
    with st.expander("Trace"):
        st.json(result["trace"])
