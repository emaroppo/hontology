"""Evaluation: runs scored, compared and broken down."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class EvaluationCalls(ApiBase):
    """``/eval``."""

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

    def live_leaderboard(
        self,
        ontology_id: int,
        manifest_path: str,
        annotator: str | None = None,
        include_partial: bool = False,
    ) -> dict:
        return self._request(
            "POST",
            "/eval/leaderboard/live",
            json={
                "ontology_id": ontology_id,
                "manifest_path": manifest_path,
                "annotator": annotator,
                "include_partial": include_partial,
            },
            timeout=300,
        )

    def run_sample(self, run_id: int, manifest_path: str, annotator: str | None = None) -> dict:
        return self._request(
            "POST",
            f"/eval/runs/{run_id}/sample",
            json={"manifest_path": manifest_path, "annotator": annotator},
        )

    def run_calendar(self, run_id: int, calendar_path: str | None = None) -> dict:
        # Every calendar window is walked: tens of seconds.
        return self._request(
            "POST",
            f"/eval/runs/{run_id}/calendar",
            json={"calendar_path": calendar_path},
            timeout=900,
        )

    def compare_arms(
        self,
        baseline: int,
        arms: list[int],
        manifest_path: str,
        annotator: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/eval/arms",
            json={
                "baseline": baseline,
                "arms": arms,
                "manifest_path": manifest_path,
                "annotator": annotator,
            },
            timeout=300,
        )

    def run_on_sample(self, run_id: int, manifest: str) -> dict:
        return self._request(
            "GET", f"/eval/runs/{run_id}/sample", params={"manifest": manifest}
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

    def run_funnel(self, run_id: int) -> dict:
        return self._request("GET", f"/eval/runs/{run_id}/funnel")

    def run_detections(self, run_id: int, *, events: bool = False) -> dict:
        return self._request(
            "GET", f"/eval/runs/{run_id}/detections", params={"events": events}
        )

    def detections_csv(self, run_id: int, *, events: bool = False) -> str:
        return (
            self._request(
                "GET", f"/eval/runs/{run_id}/detections.csv", params={"events": events}
            )
            or ""
        )
