"""Labels: the truths, the queue and adjudication, whole-document labelling, the bank."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class LabelCalls(ApiBase):
    """``/labels``."""

    def truths(self, ontology_id: int) -> dict:
        """The human label bank and each machine annotation set, with coverage."""
        return self._request("GET", "/labels/truths", params={"ontology_id": ontology_id})

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

    def pending_adjudication(self, ontology_id: int, limit: int = 50) -> list[dict]:
        return self._request(
            "GET", "/labels/pending", params={"ontology_id": ontology_id, "limit": limit}
        )

    def stale_detail(self, ontology_id: int, limit: int = 50) -> list[dict]:
        return self._request(
            "GET", "/labels/stale/detail", params={"ontology_id": ontology_id, "limit": limit}
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
        return self._request("GET", "/labels/export", params={"ontology_id": ontology_id}) or ""

    def import_labels(
        self, ontology_id: int, csv_text: str, *, overwrite: bool = False
    ) -> dict:
        return self._request(
            "POST",
            "/labels/import",
            json={"ontology_id": ontology_id, "csv": csv_text, "overwrite": overwrite},
        )
