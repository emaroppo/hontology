"""Landing page.

The app deliberately starts with nothing in it — the premise is that the ontology
is yours — so the first job of this page is to make the empty state actionable
rather than confusing.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui.client import Api

st.set_page_config(page_title="hontology", page_icon="🧭", layout="wide")

api = Api()

st.title("🧭 hontology")
st.caption(
    "Define an ontology. Detect it in a live news feed. Find out honestly how well that worked."
)

if not api.healthy():
    st.error(
        f"The API is not reachable at `{api.base_url}`.\n\n"
        "Start it with `make api`, then reload this page."
    )
    st.stop()

ontologies = api.list_ontologies()

if not ontologies:
    st.info("**No ontologies yet.** This install is empty by design.")
    st.markdown(
        "Head to the **Ontology** page to create one, or import a JSON export "
        "of one you already have."
    )
else:
    st.subheader("Ontologies")
    for ontology in ontologies:
        concepts = api.list_concepts(ontology["id"])
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{ontology['name']}** &nbsp; `{ontology['slug']}`")
            if ontology.get("description"):
                left.caption(ontology["description"])
            right.metric("Concepts", len(concepts))

st.divider()
with st.expander("How the pieces fit together"):
    st.markdown(
        """
        1. **Ontology** — describe what you care about: concepts, their
           definitions, and the inclusion and exclusion criteria that draw their
           boundaries. Criteria go into the judge prompt verbatim, so they are
           the sharpest tool you have for precision.
        2. **Ingest** — pull slices of the GDELT feed and fetch article text.
        3. **Run** — retrieve candidate `(document, concept)` pairs, then ask a
           model to judge each one.
        4. **Label** — confirm or reject pairs to build ground truth. Machine
           proposals only count once a human has adjudicated them.
        5. **Evaluate** — score retrieval and judgment *separately*, compare
           configurations, and check that a difference is bigger than the noise.
        """
    )
