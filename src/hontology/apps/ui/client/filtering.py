"""Filtering: the codebooks, class-to-code links, and what the links let through."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class FilteringCalls(ApiBase):
    """``/taxonomy``, ``/filtering`` and the corpus-wide filter walks."""

    def ingest_cameo(self) -> dict:
        return self._request("POST", "/taxonomy/cameo/ingest")

    def ingest_themes(self) -> dict:
        return self._request("POST", "/taxonomy/themes/ingest")

    def code_systems(self) -> list[dict]:
        return self._request("GET", "/taxonomy/systems")

    def ontology_links(self, ontology_id: int) -> dict[int, list[dict]]:
        links = self._request("GET", f"/taxonomy/ontologies/{ontology_id}/links")
        return {int(concept_id): rows for concept_id, rows in links.items()}

    def list_codes(self, level: str | None = None, limit: int = 500) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if level:
            params["level"] = level
        return self._request("GET", "/taxonomy/codes", params=params)

    def concept_links(self, concept_id: int) -> list[dict]:
        return self._request("GET", f"/taxonomy/concepts/{concept_id}/links")

    def concept_candidates(self, concept_id: int, run_id: int, limit: int = 20) -> list[dict]:
        return self._request(
            "GET",
            f"/taxonomy/concepts/{concept_id}/candidates",
            params={"run_id": run_id, "limit": limit},
        )

    def set_link(self, concept_id: int, code_id: int, *, linked: bool) -> None:
        self._request(
            "POST",
            "/taxonomy/links",
            json={"concept_id": concept_id, "code_id": code_id, "linked": linked},
        )

    def run_similarity(self, **payload: Any) -> dict:
        # The first run over a codebook embeds every code, which takes a while.
        return self._request("POST", "/taxonomy/similarity", json=payload, timeout=600)

    def filtering_versions(self, ontology_id: int) -> dict:
        return self._request("GET", "/filtering/versions", params={"ontology_id": ontology_id})

    def filtering_snapshot(self, ontology_id: int) -> dict:
        return self._request(
            "POST", "/filtering/versions/snapshot", params={"ontology_id": ontology_id}
        )

    def filtering_evaluate(self, part: str, **payload: Any) -> dict:
        """*part* is labels, calendar or cost; the last two take a while."""
        return self._request("POST", f"/filtering/evaluate/{part}", json=payload, timeout=900)

    # Both walk every feed record in the corpus: minutes, not seconds.

    def filter_preview(self, ontology_id: int) -> dict:
        return self._request(
            "GET", "/ingest/filter-preview", params={"ontology_id": ontology_id}, timeout=900
        )

    def filter_report(self, ontology_id: int) -> dict:
        return self._request(
            "GET", "/eval/filter-report", params={"ontology_id": ontology_id}, timeout=900
        )
