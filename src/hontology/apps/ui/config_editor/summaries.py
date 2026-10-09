"""One line per form section, for its header."""

from __future__ import annotations

from hontology.apps.ui.config_editor.state import value
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
        "ontology": f"{value('common.ontology_version')} · "
        f"{value('common.embed_body_limit'):,} characters embedded, "
        f"{value('common.judge_body_limit'):,} judged",
        "retrieval": f"{value('candidates.embed_model')} · "
        f"{value('candidates.concept_fields')} · "
        f"pool {value('candidates.pool_size')} · {cut}",
        "judge": f"{value('judge.provider')} · {value('judge.model')} · "
        f"{value('judge.prompt_id')}"
        + (" · thinking" if value("judge.think") else "")
        + (f" · {samples} samples, {value('judge.aggregation')}" if samples > 1 else ""),
        "decoding": f"temperature {value('judge.generation.temperature'):g} · context "
        f"{value('judge.generation.context_window'):,}"
        + (f" · at most {max_tokens:,} tokens" if max_tokens else "")
        + (f" · seed {seed}" if seed is not None else ""),
        "routing": value("judge.routing.provider") or "no host pinned yet",
    }
