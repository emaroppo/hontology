"""What both Judgement tabs read: the ontology, run and truth chosen in Settings,
and the per-pair prompt templates a trial can use."""

from __future__ import annotations

from dataclasses import dataclass

from hontology.apps.ui.client import Api

WORDING = {
    "definition": "Definition",
    "inclusion_criteria": "Counts when",
    "exclusion_criteria": "Does not count when",
}


@dataclass(frozen=True)
class Context:
    api: Api
    ontology: dict
    # None for human labels, else a machine annotation set's name.
    annotator: str | None
    # The run chosen in Settings, if it has judged anything.
    run: dict | None
    # Per-pair templates by prompt id.
    templates: dict[str, dict]
