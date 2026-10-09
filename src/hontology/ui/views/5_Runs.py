"""Start and watch detection runs.

Runs take minutes, so starting one returns immediately and this page polls. The
key preview is deliberately prominent: seeing that an edit reuses retrieval
*before* paying for it is the difference between iterating on a prompt in
seconds and re-embedding the corpus every time.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import config_editor, shared
from hontology.ui.client import ApiError

api, ontologies = shared.page("Runs", "⚙️")

ontology = shared.ontology(api, ontologies)

st.subheader("New run")
ontology_runs = [r for r in api.list_runs() if r["ontology_id"] == ontology["id"]]
cols = st.columns([3, 1])
start_from = cols[0].selectbox(
    "Start from",
    [None, *[r["id"] for r in ontology_runs]],
    format_func=lambda i: (
        "Defaults"
        if i is None
        else next(f"Run {r['id']} · {r['name']}" for r in ontology_runs if r["id"] == i)
    ),
    help="Fill the editor with the defaults, or with a run's config to change one thing.",
)
if cols[1].button("Load", help="Replaces what is in the editor."):
    if start_from is None:
        config_editor.load(
            api.run_options()["defaults"] | {"name": "baseline", "description": ""}
        )
    else:
        config_editor.load(api.run_config(start_from))
    st.rerun()

listing = api.versions(ontology["id"])
versions = [v["version"] for v in listing["versions"]]
config, problem = config_editor.editor(api, ontology["id"], versions)

# The stage keys the config resolves to, on the version a run would use now.
version = config.get("common", {}).get("ontology_version", "latest")
if version == "latest":
    version = listing["current"] or f"v{len(versions) + 1}"
if problem is None:
    try:
        keys = api.preview_keys(config, version)
        cols = st.columns(2)
        cols[0].metric("candidates key", keys["candidates"])
        cols[1].metric("judge key", keys["judge"].split("_")[0])
        st.caption(
            f"On ontology {version}. Two configs sharing a candidates key share retrieval: "
            "the second run copies it instead of re-embedding. Change only the prompt or "
            "the judge model and this key stays put."
        )
    except ApiError as exc:
        problem = exc.detail
        st.error(exc.detail)

st.markdown("**This run only**")
cols = st.columns(3)
document_limit = cols[0].number_input("Documents", 1, 5000, 50)
judge_limit = cols[1].number_input("Max pairs to judge", 0, 5000, 25, help="0 means no cap.")
skip_judge = cols[2].toggle("Candidates only", value=False)
if st.button("Start run", type="primary", disabled=problem is not None):
    try:
        run = api.start_run(
            ontology_id=ontology["id"],
            name=config.get("name") or "unnamed",
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
            shared.act(
                api.resume_run,
                run["id"],
                success="Resuming — pairs already judged are skipped.",
            )
