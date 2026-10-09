"""Judgement: prompts tried on a run's pairs, and a judge version scored."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class JudgementCalls(ApiBase):
    """``/judgement``."""

    # Trials are asked as the chosen run asks, and nothing is recorded.

    def judgement_templates(self) -> list[dict]:
        return self._request("GET", "/judgement/templates")

    def judgement_runs(self, ontology_id: int) -> list[dict]:
        return self._request("GET", "/judgement/runs", params={"ontology_id": ontology_id})

    def judgement_articles(
        self, run_id: int, concept_id: int, annotator: str | None = None, limit: int = 40
    ) -> list[dict]:
        return self._request(
            "POST",
            "/judgement/articles",
            json={
                "run_id": run_id,
                "concept_id": concept_id,
                "annotator": annotator,
                "limit": limit,
            },
        )

    def judgement_render(self, **trial: Any) -> dict:
        return self._request("POST", "/judgement/render", json=trial)

    def judgement_ask(self, **trial: Any) -> dict:
        # A local model can take a while over a long article.
        return self._request("POST", "/judgement/ask", json=trial, timeout=600)

    def judgement_versions(self, ontology_id: int) -> list[dict]:
        return self._request("GET", "/judgement/versions", params={"ontology_id": ontology_id})

    def judgement_evaluate(
        self, run_ids: list[int], scope: str = "responsible", annotator: str | None = None
    ) -> dict:
        return self._request(
            "POST",
            "/judgement/versions/evaluate",
            json={"run_ids": run_ids, "scope": scope, "annotator": annotator},
        )
