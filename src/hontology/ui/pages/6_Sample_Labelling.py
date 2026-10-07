"""Labelling a sample, a whole document at a time.

A labelled sample is what arms are scored against, so it is labelled blind: this
page shows the article and the leaves, never a run's verdicts, and not the
calendar stratum the document was drawn from, which names the event it was
expected to report. Documents come in the sample's frozen order, so whatever
prefix is done is itself a random sample.

Saving answers every leaf at once: the ticked ones apply, the rest do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Sample labelling", page_icon="🗂️", layout="wide")

api = Api()

st.title("🗂️ Sample labelling")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

with st.sidebar:
    manifest_path = st.text_input("Sample manifest", value="data/samples/sample-run566.json")
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read the manifest: {exc}")
        st.stop()
    # The sample is labelled against the ontology of the run it was drawn from.
    try:
        drawn_from = api.get_run(manifest["run_id"])["ontology_id"]
    except (ApiError, KeyError):
        drawn_from = None
    by_name = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    names = list(by_name)
    default = next((i for i, o in enumerate(ontologies) if o["id"] == drawn_from), 0)
    ontology = by_name[st.selectbox("Ontology", names, index=default)]
    first = st.number_input("Label the first", min_value=1, value=120, step=10)

document_ids = [entry["document_id"] for entry in manifest["order"]][: int(first)]
try:
    statuses = api.documents_status(ontology["id"], document_ids)
    leaves = api.leaves(ontology["id"])
except ApiError as exc:
    st.error(exc.detail)
    st.stop()

done = sum(1 for s in statuses if s["labelled"])
st.progress(done / len(statuses), text=f"{done} of {len(statuses)} labelled")

# Start at the first document not yet labelled; navigation moves from there.
if st.session_state.get("sample_manifest") != (manifest_path, int(first)):
    st.session_state["sample_manifest"] = (manifest_path, int(first))
    st.session_state.pop("sample_position", None)
if "sample_position" not in st.session_state:
    st.session_state["sample_position"] = next(
        (i for i, s in enumerate(statuses) if not s["labelled"]), 0
    )


def go(position: int) -> None:
    st.session_state["sample_position"] = max(0, min(position, len(statuses) - 1))


position = st.session_state["sample_position"]
nav = st.columns([1, 1, 1, 3])
if nav[0].button("← Previous", disabled=position == 0):
    go(position - 1)
    st.rerun()
if nav[1].button("Next →", disabled=position == len(statuses) - 1):
    go(position + 1)
    st.rerun()
if nav[2].button("Next unlabelled", disabled=done == len(statuses)):
    later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
    earlier = [i for i, s in enumerate(statuses) if not s["labelled"]]
    go((later or earlier)[0])
    st.rerun()
jump = nav[3].number_input(
    "Go to position", min_value=1, max_value=len(statuses), value=position + 1
)
if jump - 1 != position:
    go(jump - 1)
    st.rerun()

try:
    document = api.document_for_labelling(ontology["id"], document_ids[position])
except ApiError as exc:
    st.error(exc.detail)
    st.stop()

text_column, label_column = st.columns([3, 2])

with text_column:
    state = "labelled" if document["labelled"] else "not labelled yet"
    st.caption(f"Position {position + 1} of {len(statuses)} · {state}")
    # Many scraped articles carry no title; the text's first line usually is one.
    first_line = (document["body"] or "").strip().split("\n", 1)[0][:160]
    st.subheader(document["title"] or first_line or document["url"])
    st.markdown(f"[{document['url']}]({document['url']})")
    with st.container(height=640):
        if document["body"]:
            st.text(document["body"])
        else:
            st.warning("No text stored for this article. Read it at the link above.")

with label_column:
    st.markdown("**Which leaves does the article report?** Untick all for none.")
    current = set(document["positives"])
    key = document["document_id"]
    with st.form(f"labels_{key}"):
        chosen: list[int] = []
        family = None
        for leaf in leaves:
            if leaf["families"][0] != family:
                family = leaf["families"][0]
                st.markdown(f"**{family}**")
            explanation = "\n\n".join(
                part
                for part in (
                    leaf["definition"],
                    leaf["inclusion_criteria"] and f"Counts when: {leaf['inclusion_criteria']}",
                    leaf["exclusion_criteria"]
                    and f"Does not count when: {leaf['exclusion_criteria']}",
                )
                if part
            )
            if st.checkbox(
                leaf["name"],
                value=leaf["id"] in current,
                key=f"{key}_{leaf['id']}",
                help=explanation,
            ):
                chosen.append(leaf["id"])
        note = st.text_area("Note (optional)", value=document["note"] or "")
        saved = st.form_submit_button("Save and next", type="primary")

    if saved:
        try:
            api.label_document(ontology["id"], key, chosen, note)
        except ApiError as exc:
            st.error(exc.detail)
            st.stop()
        later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
        go(later[0] if later else position + 1)
        st.rerun()
