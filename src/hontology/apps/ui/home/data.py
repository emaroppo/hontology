"""What the Home tabs read: the scored runs, and one run's own view.

Each call is kept for five minutes: the page reruns on every click, and the
leaderboard bootstraps every run's intervals, which takes seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from hontology.apps.ui.client import Api
from hontology.apps.ui.panel import Params


@dataclass(frozen=True)
class Context:
    """What every tab reads: the client, the shared settings, the scored runs and
    the calendar scores computed this session."""

    api: Api
    params: Params
    board: dict
    calendar_cache: dict

    @property
    def runs(self) -> list[dict]:
        return self.board["rows"]

    @property
    def manifest_path(self) -> str:
        return self.params.sample

    @property
    def annotator(self) -> str | None:
        return self.params.annotator


@st.cache_data(ttl=300, show_spinner="Scoring every run on the sample…")
def scored_runs(
    ontology_id: int, manifest_path: str, annotator: str | None, partial: bool, nonce: int
) -> dict:
    """The leaderboard, kept for five minutes per ontology, sample, truth and
    partial-runs choice: it bootstraps every run's intervals, which takes seconds,
    and this page reruns on every click. *nonce* changes to recompute at once."""
    board = Api().live_leaderboard(ontology_id, manifest_path, annotator, partial)
    return board | {"computed_at": datetime.now().strftime("%H:%M:%S")}


# One run's own view: each call kept for five minutes, as the leaderboard is.


@st.cache_data(ttl=300, show_spinner=False)
def run_sample(run_id: int, manifest_path: str, annotator: str | None) -> dict:
    """One run's own sample scores, judge-only and disagreements included."""
    return Api().run_sample(run_id, manifest_path, annotator)


@st.cache_data(ttl=300, show_spinner=False)
def run_funnel(run_id: int) -> dict:
    return Api().run_funnel(run_id)


@st.cache_data(ttl=300, show_spinner=False)
def run_cutoff_report(run_id: int, annotator: str | None) -> dict:
    return Api().retrieval_report(run_id, None, annotator)["run"]


@st.cache_data(ttl=300, show_spinner=False)
def run_detections(run_id: int) -> tuple[dict, str]:
    api = Api()
    return api.run_detections(run_id), api.detections_csv(run_id)


def clear_run_caches() -> None:
    """Drop every kept one-run view, so the next look recomputes it."""
    for cached in (run_sample, run_funnel, run_cutoff_report, run_detections):
        cached.clear()
