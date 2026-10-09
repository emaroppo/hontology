"""What both Retrieval tabs read: the ontology, run and truth chosen in Settings."""

from __future__ import annotations

from dataclasses import dataclass

from hontology.apps.ui.client import Api


@dataclass(frozen=True)
class Context:
    api: Api
    ontology: dict
    # None for human labels, else a machine annotation set's name.
    annotator: str | None
    # The run chosen in Settings, if it has retrieved anything.
    run: dict | None
