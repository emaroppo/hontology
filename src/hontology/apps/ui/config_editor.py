"""A run config, edited as a form or as JSON, the same config either way.

The page holds one config. The form writes it field by field; the JSON editor
writes it whole once the text parses and the API accepts it. Switching views
carries the config across, so an edit made in one shows in the other, and JSON
that does not parse is reported rather than lost or half applied.

Every choice the form offers comes from the API (`GET /runs/options`), so a new
prompt template or aggregation rule shows up here without touching this file.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import streamlit as st

from hontology.apps.ui.client import Api, ApiError
from hontology.apps.ui.cutoff import CUTOFF_FIELDS, describe_cutoff

FORM, JSON = "Form", "JSON"
_CONFIG = "cfg:config"  # the config, last valid state
_SHOWN = "cfg:shown"  # the view drawn last, to know when to reseed widgets


def _key(path: str) -> str:
    return f"cfg:{path}"


def load(config: dict) -> None:
    """Replace the config, e.g. from a run, and redraw both views from it."""
    st.session_state[_CONFIG] = copy.deepcopy(config)
    st.session_state.pop(_SHOWN, None)


def _seed_form(config: dict) -> None:
    common, candidates, judge = config["common"], config["candidates"], config["judge"]
    generation = judge.get("generation") or {}
    routing = judge.get("routing") or {}
    values: dict[str, Any] = {
        "name": config.get("name", ""),
        "description": config.get("description", ""),
        "common.ontology_version": common["ontology_version"],
        "common.body_limit": common["body_limit"],
        "common.embed_body_limit": common.get("embed_body_limit") or common["body_limit"],
        "common.judge_body_limit": common.get("judge_body_limit") or common["body_limit"],
        "candidates.embed_provider": candidates["embed_provider"],
        "candidates.embed_model": candidates["embed_model"],
        "candidates.concept_fields": candidates["concept_fields"],
        "candidates.pool_size": candidates["pool_size"],
        "candidates.selection": candidates["selection"],
        "candidates.top_k": candidates["top_k"],
        "candidates.min_score": float(candidates["min_score"]),
        "candidates.rel_margin": float(candidates["rel_margin"]),
        "candidates.max_k": candidates["max_k"],
        "judge.provider": judge["provider"],
        "judge.model": judge["model"],
        "judge.prompt_id": judge["prompt_id"],
        "judge.think": bool(judge["think"]),
        "judge.samples": judge["samples"],
        "judge.aggregation": judge["aggregation"],
        "judge.generation.temperature": float(generation.get("temperature", 0.0)),
        "judge.generation.context_window": generation.get("context_window", 8192),
        "judge.generation.max_output_tokens": generation.get("max_output_tokens"),
        "judge.generation.seed": generation.get("seed"),
        "judge.routing.provider": routing.get("provider", ""),
        "judge.routing.quantizations": ", ".join(routing.get("quantizations") or []),
        "judge.routing.allow_fallbacks": bool(routing.get("allow_fallbacks", False)),
        "judge.routing.data_collection": routing.get("data_collection", "deny"),
    }
    for path, value in values.items():
        st.session_state[_key(path)] = value


def _value(path: str) -> Any:
    return st.session_state[_key(path)]


def _from_form() -> dict:
    judge: dict[str, Any] = {
        "provider": _value("judge.provider"),
        "model": _value("judge.model").strip(),
        "prompt_id": _value("judge.prompt_id"),
        "think": _value("judge.think"),
        "samples": int(_value("judge.samples")),
        "aggregation": _value("judge.aggregation"),
        "generation": {
            "temperature": float(_value("judge.generation.temperature")),
            "context_window": int(_value("judge.generation.context_window")),
            "max_output_tokens": _value("judge.generation.max_output_tokens"),
            "seed": _value("judge.generation.seed"),
        },
    }
    if judge["provider"] == "openrouter":
        quantizations = [
            q.strip() for q in _value("judge.routing.quantizations").split(",") if q.strip()
        ]
        judge["routing"] = {
            "provider": _value("judge.routing.provider").strip(),
            "allow_fallbacks": _value("judge.routing.allow_fallbacks"),
            "data_collection": _value("judge.routing.data_collection"),
            **({"quantizations": quantizations} if quantizations else {}),
        }
    return {
        "name": _value("name").strip() or "unnamed",
        "description": _value("description").strip(),
        "common": {
            "ontology_version": _value("common.ontology_version"),
            "body_limit": int(_value("common.body_limit")),
            "embed_body_limit": int(_value("common.embed_body_limit")),
            "judge_body_limit": int(_value("common.judge_body_limit")),
        },
        "candidates": {
            "source": "semantic",
            "embed_provider": _value("candidates.embed_provider"),
            "embed_model": _value("candidates.embed_model").strip(),
            "concept_fields": _value("candidates.concept_fields"),
            "pool_size": int(_value("candidates.pool_size")),
            "selection": _value("candidates.selection"),
            "top_k": int(_value("candidates.top_k")),
            "min_score": float(_value("candidates.min_score")),
            "rel_margin": float(_value("candidates.rel_margin")),
            "max_k": int(_value("candidates.max_k")),
        },
        "judge": judge,
    }


def _summaries() -> dict[str, str]:
    """One line per section, for its header, so a collapsed section still says
    what it holds."""
    cut = describe_cutoff({f: _value(f"candidates.{f}") for f in CUTOFF_FIELDS})
    samples = int(_value("judge.samples"))
    max_tokens = _value("judge.generation.max_output_tokens")
    seed = _value("judge.generation.seed")
    return {
        "run": _value("name") or "unnamed",
        "ontology": f"{_value('common.ontology_version')} · "
        f"{_value('common.embed_body_limit'):,} characters embedded, "
        f"{_value('common.judge_body_limit'):,} judged",
        "retrieval": f"{_value('candidates.embed_model')} · "
        f"{_value('candidates.concept_fields')} · "
        f"pool {_value('candidates.pool_size')} · {cut}",
        "judge": f"{_value('judge.provider')} · {_value('judge.model')} · "
        f"{_value('judge.prompt_id')}"
        + (" · thinking" if _value("judge.think") else "")
        + (f" · {samples} samples, {_value('judge.aggregation')}" if samples > 1 else ""),
        "decoding": f"temperature {_value('judge.generation.temperature'):g} · context "
        f"{_value('judge.generation.context_window'):,}"
        + (f" · at most {max_tokens:,} tokens" if max_tokens else "")
        + (f" · seed {seed}" if seed is not None else ""),
        "routing": _value("judge.routing.provider") or "no host pinned yet",
    }


def _section(name: str, title: str, summary: str):
    """A collapsible section, closed to start with, its header saying what it
    holds; it remembers whether it was opened."""
    return st.expander(f"**{title}** · {summary}", expanded=False, key=f"cfg:section:{name}")


def _draw_form(options: dict, versions: list[str]) -> None:
    choices = options["choices"]
    modes = {p["prompt_id"]: p["mode"] for p in choices["prompts"]}
    current_version = _value("common.ontology_version")
    version_options = ["latest", *versions]
    if current_version not in version_options:
        version_options.append(current_version)
    summary = _summaries()

    with _section("run", "Run", summary["run"]):
        cols = st.columns([1, 2])
        cols[0].text_input("Run name", key=_key("name"))
        cols[1].text_input("Description", key=_key("description"))

    with _section("ontology", "Ontology and article text", summary["ontology"]):
        cols = st.columns(4)
        cols[0].selectbox(
            "Ontology version",
            version_options,
            key=_key("common.ontology_version"),
            help="latest: the live wording, minting a version if it changed. A pinned "
            "version replays exactly the wording it names.",
        )
        cols[1].number_input(
            "Article characters", 100, 50000, step=500, key=_key("common.body_limit")
        )
        cols[2].number_input(
            "For embedding", 100, 50000, step=500, key=_key("common.embed_body_limit")
        )
        cols[3].number_input(
            "For judging", 100, 50000, step=500, key=_key("common.judge_body_limit")
        )

    with _section("retrieval", "Retrieval", summary["retrieval"]):
        cols = st.columns(4)
        cols[0].selectbox(
            "Embedding provider",
            choices["embed_providers"],
            key=_key("candidates.embed_provider"),
        )
        cols[1].text_input("Embedding model", key=_key("candidates.embed_model"))
        cols[2].selectbox(
            "Class text embedded",
            choices["concept_fields"],
            key=_key("candidates.concept_fields"),
            help="Which of a class's fields are embedded and compared with the article.",
        )
        cols[3].number_input(
            "Pool depth",
            1,
            100,
            key=_key("candidates.pool_size"),
            help="Classes ranked and stored per article, before the cutoff.",
        )
        cols = st.columns(5)
        selection = cols[0].radio(
            "Cutoff", choices["selection"], key=_key("candidates.selection")
        )
        adaptive = selection == "adaptive"
        cols[1].number_input("Top k", 1, 100, key=_key("candidates.top_k"), disabled=adaptive)
        cols[2].number_input(
            "Minimum score",
            0.0,
            1.0,
            step=0.01,
            key=_key("candidates.min_score"),
            disabled=not adaptive,
        )
        cols[3].number_input(
            "Margin below the best",
            0.0,
            1.0,
            step=0.01,
            key=_key("candidates.rel_margin"),
            disabled=not adaptive,
        )
        cols[4].number_input(
            "At most", 1, 100, key=_key("candidates.max_k"), disabled=not adaptive
        )

    with _section("judge", "Judge", summary["judge"]):
        cols = st.columns([1, 2, 2])
        provider = cols[0].selectbox(
            "Provider", choices["judge_providers"], key=_key("judge.provider")
        )
        cols[1].text_input("Model", key=_key("judge.model"))
        prompt_id = cols[2].selectbox(
            "Prompt",
            list(modes),
            key=_key("judge.prompt_id"),
            format_func=lambda p: f"{p} ({modes[p]})",
            help="per-pair: one call per article and class. per-document: one call per "
            "article. hierarchical and extract work down the class hierarchy, so they "
            "need an ontology with one.",
        )
        cols = st.columns(4)
        cols[0].toggle("Thinking", key=_key("judge.think"))
        samples = cols[1].number_input(
            "Samples",
            1,
            15,
            key=_key("judge.samples"),
            help="Asked this many times, then voted.",
        )
        cols[2].selectbox(
            "Vote",
            choices["aggregation"],
            key=_key("judge.aggregation"),
            disabled=samples == 1,
            help="How repeated samples become one verdict: majority, unanimous (precision) "
            "or any (recall).",
        )
        if modes.get(prompt_id) in ("hierarchical", "extract"):
            st.caption(
                f"{prompt_id} judges top-down, so the ontology must have a class hierarchy."
            )

    with _section("decoding", "Decoding", summary["decoding"]):
        cols = st.columns(4)
        cols[0].number_input(
            "Temperature", 0.0, 2.0, step=0.1, key=_key("judge.generation.temperature")
        )
        cols[1].number_input(
            "Context window",
            512,
            262144,
            step=1024,
            key=_key("judge.generation.context_window"),
        )
        cols[2].number_input(
            "Max output tokens",
            1,
            65536,
            value=None,
            key=_key("judge.generation.max_output_tokens"),
            placeholder="no limit",
        )
        cols[3].number_input(
            "Seed",
            0,
            2**31 - 1,
            value=None,
            key=_key("judge.generation.seed"),
            placeholder="none",
        )

    if provider == "openrouter":
        with _section("routing", "Hosted model routing", summary["routing"]):
            st.caption(
                "Hosts of one model can run it at different precisions, so the host is pinned."
            )
            cols = st.columns(4)
            cols[0].text_input(
                "Host", key=_key("judge.routing.provider"), placeholder="required"
            )
            cols[1].text_input(
                "Quantizations",
                key=_key("judge.routing.quantizations"),
                placeholder="e.g. fp8, bf16",
            )
            cols[2].toggle("Allow fallbacks", key=_key("judge.routing.allow_fallbacks"))
            cols[3].selectbox(
                "Data collection",
                choices["data_collection"],
                key=_key("judge.routing.data_collection"),
            )


def editor(api: Api, ontology_id: int, versions: list[str]) -> tuple[dict, str | None]:
    """Draw the editor; returns the config and why it cannot run, if it cannot."""
    options = api.run_options()
    if _CONFIG not in st.session_state:
        load(options["defaults"] | {"name": "baseline", "description": ""})
    # The view's one source is session state (a default too would make two), so
    # the last view chosen comes back when the page is returned to.
    if "cfg:view" not in st.session_state:
        st.session_state["cfg:view"] = st.session_state.get("keep:cfg:view", FORM)
    view = (
        st.segmented_control(
            "Edit as",
            [FORM, JSON],
            required=True,
            key="cfg:view",
            label_visibility="collapsed",
        )
        or FORM
    )
    st.session_state["keep:cfg:view"] = view
    # The fields are widget values, which Streamlit drops when the page is left;
    # the config itself is kept. So a view is redrawn from the config on entering
    # it, and also whenever its fields have been dropped.
    dropped = (
        _key("json") not in st.session_state
        if view == JSON
        else any(_key(path) not in st.session_state for path in ("name", "judge.model"))
    )
    if st.session_state.get(_SHOWN) != view or dropped:
        # Entering a view: draw it from the config as it stands.
        if view == FORM:
            _seed_form(st.session_state[_CONFIG])
        else:
            st.session_state[_key("json")] = json.dumps(st.session_state[_CONFIG], indent=2)
        st.session_state[_SHOWN] = view

    if view == FORM:
        _draw_form(options, versions)
        st.session_state[_CONFIG] = _from_form()
        problem = None
    else:
        text = st.text_area("Run config (JSON)", key=_key("json"), height=420)
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("the config must be a JSON object")
            api.preview_keys(parsed)  # the API's own validation
            st.session_state[_CONFIG] = parsed
            problem = None
        except (ValueError, ApiError) as exc:
            detail = exc.detail if isinstance(exc, ApiError) else str(exc)
            problem = f"Not applied: {detail}. Fix it, or switch to the form to drop the edit."
            st.error(problem)
    return st.session_state[_CONFIG], problem
