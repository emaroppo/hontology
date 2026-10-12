"""The config as a form: one collapsible section per part of it.

Every choice offered comes from the API's run options, so a new prompt template
or aggregation rule shows up here without touching this file.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import Api
from hontology.apps.ui.config_editor.state import key, value
from hontology.apps.ui.config_editor.summaries import summaries
from hontology.apps.ui.model_picker import DECODER, EMBEDDING, model_picker


def section(name: str, title: str, summary: str):
    """A collapsible section, closed to start with, its header saying what it
    holds; it remembers whether it was opened."""
    return st.expander(f"**{title}** · {summary}", expanded=False, key=f"cfg:section:{name}")


def draw_form(api: Api, options: dict, versions: list[str]) -> None:
    choices = options["choices"]
    current_version = value("common.ontology_version")
    version_options = ["latest", *versions]
    if current_version not in version_options:
        version_options.append(current_version)
    summary = summaries()

    with section("run", "Run", summary["run"]):
        cols = st.columns([1, 2])
        cols[0].text_input("Run name", key=key("name"))
        cols[1].text_input("Description", key=key("description"))
    with section("models", "Models", summary["models"]):
        embed_provider, provider = _model_fields(api, choices)
    with section("ontology", "Ontology and article text", summary["ontology"]):
        _ontology_fields(version_options)
    with section("retrieval", "Retrieval", summary["retrieval"]):
        _retrieval_fields(choices)
    with section("judge", "Judge", summary["judge"]):
        _judge_fields(choices)
    with section("decoding", "Decoding", summary["decoding"]):
        _decoding_fields()
    if embed_provider == "openrouter":
        with section("embed_routing", "Hosted embedding routing", summary["embed_routing"]):
            _routing_fields(choices, "candidates.embed_routing")
    if provider == "openrouter":
        with section("routing", "Hosted decoder routing", summary["routing"]):
            _routing_fields(choices, "judge.routing")


def _ontology_fields(version_options: list[str]) -> None:
    cols = st.columns(4)
    cols[0].selectbox(
        "Ontology version",
        version_options,
        key=key("common.ontology_version"),
        help="latest: the live wording, minting a version if it changed. A pinned "
        "version replays exactly the wording it names.",
    )
    cols[1].number_input(
        "Article characters", 100, 50000, step=500, key=key("common.body_limit")
    )
    cols[2].number_input(
        "For embedding", 100, 50000, step=500, key=key("common.embed_body_limit")
    )
    cols[3].number_input(
        "For judging", 100, 50000, step=500, key=key("common.judge_body_limit")
    )


def _model_fields(api: Api, choices: dict) -> tuple[str, str]:
    """The two models a run needs, picked apart; returns their two providers."""
    cols = st.columns(2)
    with cols[0]:
        embedding = model_picker(
            api,
            EMBEDDING,
            choices["embed_providers"],
            key("candidates.embed_provider"),
            key("candidates.embed_model"),
            key("candidates.embed_routing.provider"),
        )
    with cols[1]:
        decoder = model_picker(
            api,
            DECODER,
            choices["judge_providers"],
            key("judge.provider"),
            key("judge.model"),
            key("judge.routing.provider"),
        )
    return embedding[0], decoder[0]


def _retrieval_fields(choices: dict) -> None:
    cols = st.columns(2)
    cols[0].selectbox(
        "Class text embedded",
        choices["concept_fields"],
        key=key("candidates.concept_fields"),
        help="Which of a class's fields are embedded and compared with the article.",
    )
    cols[1].number_input(
        "Pool depth",
        1,
        100,
        key=key("candidates.pool_size"),
        help="Classes ranked and stored per article, before the cutoff.",
    )
    cols = st.columns(5)
    selection = cols[0].radio("Cutoff", choices["selection"], key=key("candidates.selection"))
    adaptive = selection == "adaptive"
    cols[1].number_input("Top k", 1, 100, key=key("candidates.top_k"), disabled=adaptive)
    cols[2].number_input(
        "Minimum score",
        0.0,
        1.0,
        step=0.01,
        key=key("candidates.min_score"),
        disabled=not adaptive,
    )
    cols[3].number_input(
        "Margin below the best",
        0.0,
        1.0,
        step=0.01,
        key=key("candidates.rel_margin"),
        disabled=not adaptive,
    )
    cols[4].number_input("At most", 1, 100, key=key("candidates.max_k"), disabled=not adaptive)


def _judge_fields(choices: dict) -> None:
    """The judge's fields, its model aside."""
    modes = {p["prompt_id"]: p["mode"] for p in choices["prompts"]}
    prompt_id = st.selectbox(
        "Prompt",
        list(modes),
        key=key("judge.prompt_id"),
        format_func=lambda p: f"{p} ({modes[p]})",
        help="per-pair: one call per article and class. per-document: one call per "
        "article. hierarchical and extract work down the class hierarchy, so they "
        "need an ontology with one.",
    )
    cols = st.columns(4)
    cols[0].toggle("Thinking", key=key("judge.think"))
    samples = cols[1].number_input(
        "Samples",
        1,
        15,
        key=key("judge.samples"),
        help="Asked this many times, then voted.",
    )
    cols[2].selectbox(
        "Vote",
        choices["aggregation"],
        key=key("judge.aggregation"),
        disabled=samples == 1,
        help="How repeated samples become one verdict: majority, unanimous (precision) "
        "or any (recall).",
    )
    if modes.get(prompt_id) in ("hierarchical", "extract"):
        st.caption(f"{prompt_id} judges top-down, so the ontology must have a class hierarchy.")


def _decoding_fields() -> None:
    cols = st.columns(4)
    cols[0].number_input(
        "Temperature", 0.0, 2.0, step=0.1, key=key("judge.generation.temperature")
    )
    cols[1].number_input(
        "Context window",
        512,
        262144,
        step=1024,
        key=key("judge.generation.context_window"),
    )
    cols[2].number_input(
        "Max output tokens",
        1,
        65536,
        value=None,
        key=key("judge.generation.max_output_tokens"),
        placeholder="no limit",
    )
    cols[3].number_input(
        "Seed",
        0,
        2**31 - 1,
        value=None,
        key=key("judge.generation.seed"),
        placeholder="none",
    )


def _routing_fields(choices: dict, path: str) -> None:
    """How a hosted model is routed, under *path*; its host is pinned in Models."""
    st.caption(
        "Hosts of one model can run it at different precisions, so the host is pinned "
        "(in Models) and these narrow how it may serve the model."
    )
    cols = st.columns(3)
    cols[0].text_input(
        "Quantizations",
        key=key(f"{path}.quantizations"),
        placeholder="e.g. fp8, bf16",
    )
    cols[1].toggle("Allow fallbacks", key=key(f"{path}.allow_fallbacks"))
    cols[2].selectbox(
        "Data collection",
        choices["data_collection"],
        key=key(f"{path}.data_collection"),
    )
