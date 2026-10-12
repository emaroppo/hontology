"""Run configuration and composed per-stage hashing.

A run is described by one committable JSON file, grouped into per-stage sections
(``common`` / ``candidates`` / ``judge``). Each stage's artifact key is a hash of
that stage's settings **chained to its upstream stage's key**:

    key(candidates) = hash(candidates + relevant common + ontology_version)
    key(judge)      = hash(judge + relevant common + ontology_version)
                      + "_" + key(candidates)

The chaining is what makes iteration affordable. Editing only ``judge.prompt_id``
leaves ``key(candidates)`` untouched, so the retrieval artifact is *reused*
rather than recomputed — and prompt iteration is exactly the loop you run most.
Without the split, every prompt tweak re-embeds the entire corpus.

**Behavior is hashed; infrastructure is not.** ``provider`` and ``model`` change
what comes out, so they belong in the key. Hosts, ports, connection URLs and API
keys are transport: the same run served from a different machine is the *same
run*. Hashing them would fork the artifact tree on every environment change and
destroy the ability to compare results across machines — so they are excluded
here and recorded in the run manifest instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Fields that describe *where* things run rather than *what* runs. Never hashed.
INFRA_KEYS = frozenset(
    {
        "db_url",
        "database_url",
        "ollama_host",
        "llamacpp_host",
        "llamacpp_embed_host",
        "embed_host",
        "api_key",
        "region",
        "base_url",
    }
)

COMMON_DEFAULTS: dict[str, Any] = {
    "ontology_version": "latest",
    # Character budgets applied before the text reaches a model. Splitting them
    # matters: retrieval and judging want very different amounts of an article.
    "body_limit": 3000,
    "embed_body_limit": None,  # inherits body_limit when None
    "judge_body_limit": None,
}

CANDIDATES_DEFAULTS: dict[str, Any] = {
    "source": "semantic",
    "embed_provider": "ollama",
    "embed_model": "nomic-embed-text",
    "concept_fields": "name+definition",
    "selection": "adaptive",  # "top-k" | "adaptive"
    "top_k": 3,
    "min_score": 0.45,
    "rel_margin": 0.05,
    "max_k": 8,
    # Retrieval depth before the cutoff is applied. The pool is stored so
    # ranking quality can be scored separately from cutoff quality.
    "pool_size": 20,
}

GENERATION_DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "context_window": 8192,
    "max_output_tokens": None,
    "seed": None,
}

JUDGE_DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "model": "huihui_ai/qwen3.5-abliterated:9b",
    "prompt_id": "strict_v1",
    "think": False,
    # Several samples plus a majority vote. The vote fraction is a more honest
    # confidence than the model's self-report.
    "samples": 1,
    "aggregation": "majority",
    "generation": dict(GENERATION_DEFAULTS),
}

# How repeated samples resolve to one verdict. Kept here so the config layer can
# reject an unknown value without importing the judge.
AGGREGATIONS = ("majority", "unanimous", "any")


class ConfigError(ValueError):
    pass


def _merge(defaults: dict, override: dict | None) -> dict:
    out = dict(defaults)
    for key, value in (override or {}).items():
        if key in INFRA_KEYS:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def normalize(config: dict) -> dict:
    """Return a fully-defaulted config, with inherited limits resolved."""
    common = _merge(COMMON_DEFAULTS, config.get("common"))
    candidates = _merge(CANDIDATES_DEFAULTS, config.get("candidates"))
    judge = _merge(JUDGE_DEFAULTS, config.get("judge"))

    if candidates["source"] not in ("semantic",):
        raise ConfigError(f"unknown candidates.source {candidates['source']!r}")
    if candidates["selection"] not in ("top-k", "adaptive"):
        raise ConfigError(f"unknown candidates.selection {candidates['selection']!r}")
    if judge["samples"] < 1:
        raise ConfigError("judge.samples must be at least 1")
    # Validated here rather than at use, so a value that would do nothing is
    # rejected when the config is written instead of being silently ignored.
    if judge["aggregation"] not in AGGREGATIONS:
        raise ConfigError(
            f"unknown judge.aggregation {judge['aggregation']!r}; "
            f"expected one of {AGGREGATIONS}"
        )
    _check_routing(judge, "provider", "routing", "judge.routing")
    _check_routing(candidates, "embed_provider", "embed_routing", "candidates.embed_routing")

    # A single body_limit is the usual knob; the per-stage ones override it.
    base = common["body_limit"]
    if common.get("embed_body_limit") is None:
        common["embed_body_limit"] = base
    if common.get("judge_body_limit") is None:
        common["judge_body_limit"] = base

    return {
        "name": config.get("name", "unnamed"),
        "description": config.get("description", ""),
        "common": common,
        "candidates": candidates,
        "judge": judge,
    }


ROUTING_KEYS = {"provider", "quantizations", "allow_fallbacks", "data_collection"}


def _check_routing(section: dict, provider_key: str, routing_key: str, label: str) -> None:
    """A hosted model, decoder or embedding, must say which host serves it;
    nothing else may route.

    Routing is behavior: hosts of one model run it at different precisions, and
    their answers, or vectors, can differ. It is absent from configs that do not
    use it, so their hashes are unchanged.
    """
    routing = section.get(routing_key)
    if section[provider_key] == "openrouter":
        if not routing or not routing.get("provider"):
            raise ConfigError(
                f"{label}.provider is required for openrouter: pin the host, "
                "or calls in one run may be served by different setups"
            )
    elif routing is not None:
        raise ConfigError(f"{label} applies only to openrouter, not {section[provider_key]!r}")
    if routing is not None:
        unknown = set(routing) - ROUTING_KEYS
        if unknown:
            raise ConfigError(f"unknown {label} keys: {sorted(unknown)}")
        if routing.get("data_collection", "deny") not in ("deny", "allow"):
            raise ConfigError(f"{label}.data_collection must be deny or allow")


def stable_hash(payload: Any, n: int = 12) -> str:
    """The first *n* hex digits of a SHA-256 over *payload*'s canonical JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:n]


def stage_keys(normalized: dict, ontology_version: str) -> dict[str, str]:
    """Composed per-stage artifact keys.

    ``ontology_version`` is the resolved concrete version (``vN``), never
    ``"latest"`` — otherwise two runs months apart would share a key while having
    been scored against different concept wording.
    """
    common = normalized["common"]

    candidates_key = stable_hash(
        {
            "ontology_version": ontology_version,
            # Only the limit that actually affects retrieval.
            "embed_body_limit": common["embed_body_limit"],
            "candidates": normalized["candidates"],
        }
    )
    retrieval_key = _retrieval_key(normalized, candidates_key)
    judge_key = stable_hash(
        {
            "ontology_version": ontology_version,
            "judge_body_limit": common["judge_body_limit"],
            "judge": normalized["judge"],
        }
    )
    # Chained: the judge artifact is only meaningful for what it was shown.
    return {
        "candidates": candidates_key,
        "retrieval": retrieval_key,
        "judge": f"{judge_key}_{retrieval_key}",
    }


def _retrieval_key(normalized: dict, candidates_key: str) -> str:
    """What picks the leaves the judge is shown.

    For a flat prompt that is the candidates alone, so its key is theirs and
    flat runs keep their keys. A top-down prompt adds its routing, extraction
    and ranking wording, the decoder that routes and how it decodes, and the
    article length routing reads.
    """
    from hontology.pipeline.judge import prompts
    from hontology.pipeline.runs.fingerprints import retrieval_fingerprint

    judge = normalized["judge"]
    if prompts.get(judge["prompt_id"]).mode not in prompts.TOP_DOWN_MODES:
        return candidates_key
    decoder = {k: judge.get(k) for k in ("provider", "model", "routing", "generation", "think")}
    return stable_hash(
        {
            "candidates": candidates_key,
            "prompt": retrieval_fingerprint(judge["prompt_id"]),
            "decoder": decoder,
            "judge_body_limit": normalized["common"]["judge_body_limit"],
        }
    )


def manifest(normalized: dict, ontology_version: str, *, infra: dict[str, Any]) -> dict:
    """Everything needed to explain a run after the fact.

    Infrastructure lives here rather than in the key: recorded for provenance,
    excluded from identity.
    """
    return {
        "name": normalized["name"],
        "description": normalized["description"],
        "ontology_version": ontology_version,
        "config": normalized,
        "keys": stage_keys(normalized, ontology_version),
        "infra": infra,
    }
