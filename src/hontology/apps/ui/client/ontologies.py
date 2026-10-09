"""Ontologies and their classes: authoring, import and export, versions, lint."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class OntologyCalls(ApiBase):
    """``/ontologies``."""

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

    def hierarchy(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/hierarchy")

    def export_ontology(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/export")

    def import_ontology(self, payload: dict) -> dict:
        return self._request("POST", "/ontologies/import", json=payload)

    def export_owl(self, ontology_id: int) -> str:
        return self._request("GET", f"/ontologies/{ontology_id}/export.owl")

    def import_owl(self, turtle: str, *, allow_text_change: bool = False) -> dict:
        return self._request(
            "POST",
            "/ontologies/import-owl",
            json={"turtle": turtle, "allow_text_change": allow_text_change},
        )

    def versions(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/versions")

    def lint_ontology(self, ontology_id: int) -> dict:
        return self._request("GET", f"/ontologies/{ontology_id}/lint")
