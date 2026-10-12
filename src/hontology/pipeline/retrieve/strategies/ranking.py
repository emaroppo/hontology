"""Ranking leaves by how close their embedding is to an event's."""

from __future__ import annotations

import numpy as np

from hontology.db.models import Concept
from hontology.pipeline.retrieve import embed, model_text


class LeafRanker:
    """Orders leaves by how close their embedding is to an event's."""

    def __init__(self, provider, model: str, leaves: list[Concept], k: int) -> None:
        self.provider, self.model, self.k = provider, model, k
        self.ids = [c.id for c in leaves]
        prefix = model_text.document_prefix(model)
        self.vectors = self._unit(
            [prefix + embed.concept_text(c, "name+definition") for c in leaves]
        )

    def _unit(self, texts: list[str]) -> np.ndarray:
        texts = [model_text.normalize_for_embedding(t) for t in texts]
        vectors = np.array(self.provider.embed(texts, model=self.model), dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def __call__(self, event: dict, allowed: set[int]) -> list[int]:
        text = f"{event['description']}\n{event['evidence']}".strip()
        scores = self.vectors @ self._unit([model_text.query_prefix(self.model) + text])[0]
        ranked = sorted(
            (i for i in range(len(self.ids)) if self.ids[i] in allowed),
            key=lambda i: (-scores[i], self.ids[i]),
        )
        return [self.ids[i] for i in ranked[: self.k]]
