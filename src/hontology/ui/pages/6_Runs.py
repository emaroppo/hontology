"""Start and watch detection runs.

Runs take minutes, so starting one returns immediately and this page polls. The
key preview is deliberately prominent: seeing that an edit reuses retrieval
*before* paying for it is the difference between iterating on a prompt in
seconds and re-embedding the corpus every time.
"""

from __future__ import annotations

import json

import streamlit as st

from hontology.ui import shared
from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Runs", page_icon="⚙️", layout="wide")

api = Api()

st.title("⚙️ Runs")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

DEFAULT_CONFIG = {
    "candidates": {
        "source": "semantic",
        "selection": "adaptive",
        "min_score": 0.5,
        "rel_margin": 0.05,
        "max_k": 3,
        "pool_size": 8,
    },
    "judge": {"prompt_id": "strict_v1", "think": False, "samples": 1},
    "common": {"body_limit": 2500},
}

ontology = shared.ontology(api, ontologies)

st.subheader("Configuration")
cols = st.columns(4)
name = cols[0].text_input("Run name", "baseline")
document_limit = cols[1].number_input("Documents", 1, 5000, 50)
judge_limit = cols[2].number_input("Max pairs to judge", 0, 5000, 25, help="0 means no cap.")
skip_judge = cols[3].toggle("Candidates only", value=False)
config_text = st.text_area(
    "Run config (JSON)", json.dumps(DEFAULT_CONFIG, indent=2), height=280
)

try:
    config = json.loads(config_text)
    config_error = None
except json.JSONDecodeError as exc:
    config, config_error = None, str(exc)

if config_error:
    st.error(f"Invalid JSON: {config_error}")
else:
    try:
        keys = api.preview_keys(config)
        cols = st.columns(2)
        cols[0].metric("candidates key", keys["candidates"])
        cols[1].metric("judge key", keys["judge"].split("_")[0])
        st.caption(
            "Two configs sharing a candidates key share retrieval — the second run "
            "copies it instead of re-embedding. Change only the prompt or the judge "
            "model and this key stays put."
        )
    except ApiError as exc:
        st.error(exc.detail)

    if st.button("Start run", type="primary"):
        try:
            run = api.start_run(
                ontology_id=ontology["id"],
                name=name,
                config=config,
                document_limit=int(document_limit),
                judge_limit=int(judge_limit) or None,
                skip_judge=skip_judge,
            )
            st.session_state["watch_run"] = run["id"]
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)

st.divider()
st.subheader("Runs")

try:
    runs = api.list_runs()
except ApiError as exc:
    st.error(exc.detail)
    st.stop()

watching = st.session_state.get("watch_run")
active = [r for r in runs if r["status"] in ("pending", "running")]

if active:
    st.caption(f"{len(active)} run(s) in flight — this page refreshes while they work.")
    for run in active:
        with st.container(border=True):
            st.markdown(
                f"**{run['id']}: {run['name']}** — {run['status']} / {run['stage'] or '…'}"
            )
            total = run["progress_total"] or 0
            done = run["progress_done"] or 0
            st.progress(done / total if total else 0.0, text=f"{done}/{total} pairs judged")
    # Cheap poll: a rerun re-fetches, which is enough for a minutes-long job.
    if st.button("Refresh"):
        st.rerun()

for run in runs[:25]:
    icon = {"done": "✅", "failed": "❌", "running": "⏳", "pending": "⏳"}.get(
        run["status"], "•"
    )
    with st.expander(
        f"{icon} {run['id']}: {run['name']} — {run['status']}"
        + (" ← just started" if run["id"] == watching else "")
    ):
        cols = st.columns(3)
        cols[0].markdown(f"ontology version `{run['ontology_version']}`")
        cols[1].markdown(f"candidates `{run['candidates_key']}`")
        cols[2].markdown(f"judge `{run['judge_key'].split('_')[0]}`")

        if run["error"]:
            st.error(run["error"])

        resumable = run["status"] in ("done", "failed", "candidates")
        if resumable and st.button("Resume", key=f"resume_{run['id']}"):
            try:
                api.resume_run(run["id"])
                st.success("Resuming — pairs already judged are skipped.")
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
