"""The config the editor holds, and its form fields in session state.

The config itself is kept under one key; the form's fields are widget values,
one per config path, seeded from the config on entering the form and read back
into it after every run.
"""

from __future__ import annotations

import copy
from typing import Any

import streamlit as st

CONFIG = "cfg:config"  # the config, last valid state
SHOWN = "cfg:shown"  # the view drawn last, to know when to reseed widgets


def key(path: str) -> str:
    return f"cfg:{path}"


def load(config: dict) -> None:
    """Replace the config, e.g. from a run, and redraw both views from it."""
    st.session_state[CONFIG] = copy.deepcopy(config)
    st.session_state.pop(SHOWN, None)


def seed_form(config: dict) -> None:
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
        st.session_state[key(path)] = value


def value(path: str) -> Any:
    return st.session_state[key(path)]


def from_form() -> dict:
    judge: dict[str, Any] = {
        "provider": value("judge.provider"),
        "model": value("judge.model").strip(),
        "prompt_id": value("judge.prompt_id"),
        "think": value("judge.think"),
        "samples": int(value("judge.samples")),
        "aggregation": value("judge.aggregation"),
        "generation": {
            "temperature": float(value("judge.generation.temperature")),
            "context_window": int(value("judge.generation.context_window")),
            "max_output_tokens": value("judge.generation.max_output_tokens"),
            "seed": value("judge.generation.seed"),
        },
    }
    if judge["provider"] == "openrouter":
        quantizations = [
            q.strip() for q in value("judge.routing.quantizations").split(",") if q.strip()
        ]
        judge["routing"] = {
            "provider": value("judge.routing.provider").strip(),
            "allow_fallbacks": value("judge.routing.allow_fallbacks"),
            "data_collection": value("judge.routing.data_collection"),
            **({"quantizations": quantizations} if quantizations else {}),
        }
    return {
        "name": value("name").strip() or "unnamed",
        "description": value("description").strip(),
        "common": {
            "ontology_version": value("common.ontology_version"),
            "body_limit": int(value("common.body_limit")),
            "embed_body_limit": int(value("common.embed_body_limit")),
            "judge_body_limit": int(value("common.judge_body_limit")),
        },
        "candidates": {
            "source": "semantic",
            "embed_provider": value("candidates.embed_provider"),
            "embed_model": value("candidates.embed_model").strip(),
            "concept_fields": value("candidates.concept_fields"),
            "pool_size": int(value("candidates.pool_size")),
            "selection": value("candidates.selection"),
            "top_k": int(value("candidates.top_k")),
            "min_score": float(value("candidates.min_score")),
            "rel_margin": float(value("candidates.rel_margin")),
            "max_k": int(value("candidates.max_k")),
        },
        "judge": judge,
    }
