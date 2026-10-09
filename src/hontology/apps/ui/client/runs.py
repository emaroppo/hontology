"""Runs: start, watch and resume one, and the config it is made from."""

from __future__ import annotations

from typing import Any

from hontology.apps.ui.client.base import ApiBase


class RunCalls(ApiBase):
    """``/runs``."""

    def start_run(self, **payload: Any) -> dict:
        return self._request("POST", "/runs", json=payload)

    def get_run(self, run_id: int) -> dict:
        return self._request("GET", f"/runs/{run_id}")

    def resume_run(self, run_id: int, judge_limit: int | None = None) -> dict:
        params = {"judge_limit": judge_limit} if judge_limit else {}
        return self._request("POST", f"/runs/{run_id}/resume", params=params)

    def run_options(self) -> dict:
        return self._request("GET", "/runs/options")

    def run_config(self, run_id: int) -> dict:
        return self._request("GET", f"/runs/{run_id}/config")

    def preview_keys(self, config: dict, ontology_version: str = "v1") -> dict:
        return self._request(
            "POST",
            "/runs/keys",
            json={"config": config, "ontology_version": ontology_version},
        )
