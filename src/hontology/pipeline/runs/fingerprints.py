"""Fingerprints of the code and prompt text in force, recorded on a run when it is created."""

from __future__ import annotations

import functools
import inspect
from typing import Any

from hontology.db.models import (
    Concept,
    Document,
    Run,
)
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs.config import stable_hash


@functools.cache
def retrieval_code_hash() -> str:
    """The code that turns embeddings into a ranking and a ranking into a cut."""
    from hontology.pipeline.retrieve import candidates, embed, model_text, tuning

    parts = [
        inspect.getsource(candidates.select_adaptive),
        inspect.getsource(candidates.build_semantic),
        candidates._NEAREST_CONCEPTS.text,
        inspect.getsource(embed.concept_text),
        inspect.getsource(model_text.normalize_for_embedding),
        inspect.getsource(model_text._prefixes),
        tuning._POOL.text,
    ]
    return stable_hash(parts, 12)


@functools.cache
def prompt_fingerprint(prompt_id: str) -> str:
    """A hash of everything a template sends, rendered on fixed inputs.

    Rendering rather than hashing source catches an edit to a shared constant
    (a response shape, say) that a builder only refers to.
    """
    from hontology.pipeline.judge import prompts

    template = prompts.get(prompt_id)
    document = Document(url="https://example.test/a", title="Title")
    concepts = [
        Concept(id=1, name="Alpha", definition="D1", inclusion_criteria="I1"),
        Concept(id=2, name="Beta", definition="D2", exclusion_criteria="E2"),
    ]
    event = {"description": "An event.", "country": "XX", "quote": "A quote."}
    parts: list[Any] = [template.prompt_id, template.mode]
    for name, value in vars(template).items():
        if isinstance(value, str | int) or value is None:
            parts.append((name, value))
            continue
        attempts = (
            lambda f: f(document, concepts[0], "BODY", 100),
            lambda f: f(document, concepts, "BODY", 100),
            lambda f: f(event, concepts),
            lambda f: f(event, concepts, True),
        )
        for attempt in attempts:
            try:
                parts.append((name, attempt(value)))
                break
            except Exception:  # noqa: BLE001, S112 - try the next call shape
                continue
        else:
            parts.append((name, inspect.getsource(value)))
    return stable_hash(parts, 12)


def recorded(run: Run) -> dict:
    """Fingerprints to stamp on a run at creation."""
    judge = run_config.normalize(run.config or {})["judge"]
    return {
        "retrieval_code": retrieval_code_hash(),
        "prompt": prompt_fingerprint(judge["prompt_id"]),
    }
