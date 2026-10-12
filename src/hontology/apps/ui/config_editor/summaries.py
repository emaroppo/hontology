"""One line per form section, for its header."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.config_editor.state import key, value
from hontology.apps.ui.cutoff import CUTOFF_FIELDS, describe_cutoff


def summaries() -> dict[str, str]:
    """One line per section, for its header, so a collapsed section still says
    what it holds."""
    cut = describe_cutoff({f: value(f"candidates.{f}") for f in CUTOFF_FIELDS})
    samples = int(value("judge.samples"))
    max_tokens = value("judge.generation.max_output_tokens")
    seed = value("judge.generation.seed")
    return {
        "run": value("name") or "unnamed",
        "models": f"embedding {value('candidates.embed_provider')} · "
        f"{value('candidates.embed_model')}; decoder {value('judge.provider')} · "
        f"{value('judge.model')}",
        "ontology": f"{value('common.ontology_version')} · "
        f"{value('common.embed_body_limit'):,} characters embedded, "
        f"{value('common.judge_body_limit'):,} judged",
        "retrieval": f"{value('candidates.concept_fields')} · "
        f"pool {value('candidates.pool_size')} · {cut}",
        "judge": f"{value('judge.prompt_id')}"
        + (" · thinking" if value("judge.think") else "")
        + (f" · {samples} samples, {value('judge.aggregation')}" if samples > 1 else ""),
        "decoding": f"temperature {value('judge.generation.temperature'):g} · context "
        f"{value('judge.generation.context_window'):,}"
        + (f" · at most {max_tokens:,} tokens" if max_tokens else "")
        + (f" · seed {seed}" if seed is not None else ""),
        "routing": _host("judge.routing"),
        "embed_routing": _host("candidates.embed_routing"),
    }


def _host(path: str) -> str:
    """The pinned host under *path*, which the Models section may not have drawn."""
    host = st.session_state.get(key(f"{path}.provider"))
    return f"host {host}" if host else "no host pinned yet"
