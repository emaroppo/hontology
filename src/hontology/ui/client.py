"""Thin HTTP client for the API.

The UI has no database access at all. Everything it shows or changes goes through
here, which is why the Streamlit layer can be replaced without moving any logic
out of it.
"""

from __future__ import annotations

from typing import Any

import httpx

from hontology.config import get_settings


class ApiError(RuntimeError):
    """A non-2xx response, carrying the API's own message where it gave one."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Api:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or get_settings().api_base_url).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, f"cannot reach the API at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("detail", detail)
                if isinstance(detail, list):  # pydantic validation errors
                    detail = "; ".join(
                        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}"
                        for e in detail
                    )
            except ValueError:
                pass
            raise ApiError(response.status_code, str(detail))

        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # --- meta ---------------------------------------------------------------

    def healthy(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "ok"
        except ApiError:
            return False

    # --- ontologies ---------------------------------------------------------

    def list_ontologies(self) -> list[dict]:
        return self._request("GET", "/ontologies")

    def create_ontology(self, *, slug: str, name: str, description: str | None) -> dict:
        return self._request(
            "POST",
            "/ontologies",
            json={"slug": slug, "name": name, "description": description},
        )

    def delete_ontology(self, ontology_id: int) -> None:
        self._request("DELETE", f"/ontologies/{ontology_id}")

    # --- concepts -----------------------------------------------------------

    def list_concepts(self, ontology_id: int) -> list[dict]:
        return self._request("GET", f"/ontologies/{ontology_id}/concepts")

    def create_concept(self, ontology_id: int, **fields: Any) -> dict:
        return self._request("POST", f"/ontologies/{ontology_id}/concepts", json=fields)

    def update_concept(self, ontology_id: int, concept_id: int, **fields: Any) -> dict:
        return self._request(
            "PATCH", f"/ontologies/{ontology_id}/concepts/{concept_id}", json=fields
        )

    def delete_concept(self, ontology_id: int, concept_id: int) -> None:
        self._request("DELETE", f"/ontologies/{ontology_id}/concepts/{concept_id}")

    # --- portability & versioning -------------------------------------------

    def export_ontology(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/export")

    def import_ontology(self, payload: dict) -> dict:
        return self._request("POST", "/ontologies/import", json=payload)

    def snapshot(self, ontology_id: int) -> dict:
        return self._request("POST", f"/ontologies/{ontology_id}/snapshot")

    # --- taxonomy -----------------------------------------------------------

    def ingest_cameo(self) -> dict:
        return self._request("POST", "/taxonomy/cameo/ingest")

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
        return self._request("POST", "/taxonomy/similarity", json=payload)

    # --- labels -------------------------------------------------------------

    def labelling_queue(
        self,
        ontology_id: int,
        *,
        limit: int = 50,
        per_concept_cap: int | None = None,
        include_unjudged: bool = True,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "ontology_id": ontology_id,
            "limit": limit,
            "include_unjudged": include_unjudged,
        }
        if per_concept_cap:
            params["per_concept_cap"] = per_concept_cap
        return self._request("GET", "/labels/queue", params=params)

    def label_stats(self, ontology_id: int) -> dict:
        return self._request("GET", "/labels/stats", params={"ontology_id": ontology_id})

    def create_label(self, **payload: Any) -> dict:
        return self._request("POST", "/labels", json=payload)

    def adjudicate_label(
        self, label_id: int, *, matched: bool, note: str | None = None
    ) -> dict:
        return self._request(
            "POST", f"/labels/{label_id}/adjudicate", json={"matched": matched, "note": note}
        )

    # --- evaluation ---------------------------------------------------------

    def list_runs(self) -> list[dict]:
        return self._request("GET", "/eval/runs")

    def evaluate_run(
        self, run_id: int, *, include_machine: bool = False, include_stale: bool = False
    ) -> dict:
        return self._request(
            "GET",
            f"/eval/runs/{run_id}",
            params={"include_machine": include_machine, "include_stale": include_stale},
        )

    def compare_runs(self, run_a: int, run_b: int, *, include_machine: bool = False) -> dict:
        return self._request(
            "GET",
            "/eval/compare",
            params={"run_a": run_a, "run_b": run_b, "include_machine": include_machine},
        )

    def leaderboard(self, limit: int = 50) -> list[dict]:
        return self._request("GET", "/eval/leaderboard", params={"limit": limit})

    def pending_adjudication(self, ontology_id: int, limit: int = 50) -> list[dict]:
        return self._request(
            "GET", "/labels/pending", params={"ontology_id": ontology_id, "limit": limit}
        )

    def stale_detail(self, ontology_id: int, limit: int = 50) -> list[dict]:
        return self._request(
            "GET", "/labels/stale/detail", params={"ontology_id": ontology_id, "limit": limit}
        )

    # --- runs ---------------------------------------------------------------

    def start_run(self, **payload: Any) -> dict:
        return self._request("POST", "/runs", json=payload)

    def get_run(self, run_id: int) -> dict:
        return self._request("GET", f"/runs/{run_id}")

    def resume_run(self, run_id: int, judge_limit: int | None = None) -> dict:
        params = {"judge_limit": judge_limit} if judge_limit else {}
        return self._request("POST", f"/runs/{run_id}/resume", params=params)

    def preview_keys(self, config: dict, ontology_version: str = "v1") -> dict:
        return self._request(
            "POST",
            "/runs/keys",
            json={"config": config, "ontology_version": ontology_version},
        )

    def leaves(self, ontology_id: int) -> list[dict]:
        return self._request("GET", "/labels/leaves", params={"ontology_id": ontology_id})

    def documents_status(self, ontology_id: int, document_ids: list[int]) -> list[dict]:
        return self._request(
            "POST",
            "/labels/documents/status",
            json={"ontology_id": ontology_id, "document_ids": document_ids},
        )

    def document_for_labelling(self, ontology_id: int, document_id: int) -> dict:
        return self._request(
            "GET", f"/labels/documents/{document_id}", params={"ontology_id": ontology_id}
        )

    def label_document(
        self, ontology_id: int, document_id: int, concept_ids: list[int], note: str | None
    ) -> dict:
        return self._request(
            "PUT",
            f"/labels/documents/{document_id}",
            json={"ontology_id": ontology_id, "concept_ids": concept_ids, "note": note},
        )

    def export_labels(self, ontology_id: int) -> str:
        response = httpx.get(
            f"{self.base_url}/labels/export",
            params={"ontology_id": ontology_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def import_labels(
        self, ontology_id: int, csv_text: str, *, overwrite: bool = False
    ) -> dict:
        return self._request(
            "POST",
            "/labels/import",
            json={"ontology_id": ontology_id, "csv": csv_text, "overwrite": overwrite},
        )

    def run_breakdown(
        self, run_id: int, *, dimension: str = "concept", include_machine: bool = False
    ) -> list[dict]:
        return self._request(
            "GET",
            f"/eval/runs/{run_id}/breakdown",
            params={"dimension": dimension, "include_machine": include_machine},
        )

    def run_errors(
        self, run_id: int, *, kind: str | None = None, include_machine: bool = False
    ) -> dict:
        params: dict[str, Any] = {"include_machine": include_machine}
        if kind:
            params["kind"] = kind
        return self._request("GET", f"/eval/runs/{run_id}/errors", params=params)

    def lint_ontology(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/lint")

    def filter_report(self, ontology_id: int) -> dict:
        return self._request("GET", "/eval/filter-report", params={"ontology_id": ontology_id})

    def run_funnel(self, run_id: int) -> dict:
        return self._request("GET", f"/eval/runs/{run_id}/funnel")

    def run_detections(self, run_id: int, *, events: bool = False) -> dict:
        return self._request(
            "GET", f"/eval/runs/{run_id}/detections", params={"events": events}
        )

    def detections_csv(self, run_id: int, *, events: bool = False) -> str:
        response = httpx.get(
            f"{self.base_url}/eval/runs/{run_id}/detections.csv",
            params={"events": events},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text
