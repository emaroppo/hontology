"""Retrieval: a run's ranking, a cutoff through it, and a version scored live."""

from __future__ import annotations

from hontology.apps.ui.client.base import ApiBase


class RetrievalCalls(ApiBase):
    """``/retrieval``."""

    # Each call carries the cutoff (None: the run's own) and the truth: None for
    # human labels, else a machine annotation set's name.

    def retrieval_runs(self, ontology_id: int) -> list[dict]:
        return self._request("GET", "/retrieval/runs", params={"ontology_id": ontology_id})

    def retrieval_report(
        self, run_id: int, cutoff: dict | None, annotator: str | None = None
    ) -> dict:
        return self._request(
            "POST",
            f"/retrieval/runs/{run_id}/report",
            json={"cutoff": cutoff, "annotator": annotator},
        )

    def retrieval_labelled(self, run_id: int, annotator: str | None = None) -> list[dict]:
        return self._request(
            "POST", f"/retrieval/runs/{run_id}/labelled", json={"annotator": annotator}
        )

    def retrieval_document(
        self, run_id: int, document_id: int, cutoff: dict | None, annotator: str | None = None
    ) -> dict:
        return self._request(
            "POST",
            f"/retrieval/runs/{run_id}/documents/{document_id}",
            json={"cutoff": cutoff, "annotator": annotator},
        )

    def retrieval_concept(
        self,
        run_id: int,
        concept_id: int,
        cutoff: dict | None,
        annotator: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        return self._request(
            "POST",
            f"/retrieval/runs/{run_id}/concepts/{concept_id}",
            params={"limit": limit},
            json={"cutoff": cutoff, "annotator": annotator},
        )

    def retrieval_text(
        self,
        run_id: int,
        text: str | None,
        cutoff: dict | None = None,
        pool_size: int = 20,
        *,
        document_id: int | None = None,
        embedding: tuple[str, ...] | None = None,
    ) -> dict:
        """Pasted text, or a corpus article, ranked live as *run_id*'s retrieval
        ranks an article, with *embedding* in place of its own if given."""
        return self._request(
            "POST",
            "/retrieval/text",
            json={
                "run_id": run_id,
                "text": text,
                "document_id": document_id,
                "cutoff": cutoff,
                "pool_size": pool_size,
                "embedding": embedding,
            },
            timeout=300,
        )

    def retrieval_strategies(self) -> list[dict]:
        return self._request("GET", "/retrieval/strategies")

    def retrieval_strategy(
        self,
        run_id: int,
        prompt_id: str,
        cutoff: dict,
        *,
        text: str | None = None,
        document_id: int | None = None,
        embedding: tuple[str, ...] | None = None,
        decoder: tuple[str, ...] | None = None,
    ) -> dict:
        """A prompt's retrieval run on one article or pasted text; not stored."""
        return self._request(
            "POST",
            "/retrieval/strategy",
            json={
                "run_id": run_id,
                "prompt_id": prompt_id,
                "cutoff": cutoff,
                "text": text,
                "document_id": document_id,
                "embedding": embedding,
                "decoder": decoder,
            },
            timeout=600,
        )

    def retrieval_versions(self, ontology_id: int) -> list[dict]:
        return self._request("GET", "/retrieval/versions", params={"ontology_id": ontology_id})

    def retrieval_evaluate(
        self, run_id: int, cutoff: dict, pool_size: int = 20, annotator: str | None = None
    ) -> dict:
        return self._request(
            "POST",
            "/retrieval/versions/evaluate",
            json={
                "run_id": run_id,
                "cutoff": cutoff,
                "pool_size": pool_size,
                "annotator": annotator,
            },
        )
