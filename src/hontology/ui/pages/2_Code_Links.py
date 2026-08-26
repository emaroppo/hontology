"""Review concept↔code associations.

Two things this page must make unmistakable, because getting either wrong costs
real curation work:

- which links are machine proposals and which a human asserted, and
- that ticking a proposal *promotes* it, so the next recompute cannot drop it.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Code links", page_icon="🔗", layout="wide")

api = Api()

st.title("🔗 Code links")
st.caption(
    "The `code` retrieval strategy reaches your concepts through these links. "
    "It is only meaningful for an ontology whose concepts map onto CAMEO at all — "
    "for anything else, use semantic retrieval instead."
)

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

with st.sidebar:
    st.subheader("Codebook")
    if st.button("Ingest CAMEO codebook", help="Fetches the public lookup. Idempotent."):
        try:
            result = api.ingest_cameo()
            st.success(
                f"{result['total']} codes "
                f"({result['inserted']} new, {result['updated']} updated)."
            )
        except ApiError as exc:
            st.error(exc.detail)

    labels = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    ontology = labels[st.selectbox("Ontology", list(labels))]

    st.divider()
    st.subheader("Propose links")
    level = st.selectbox("Code level", ["event", "base", "root"])
    adaptive = st.toggle(
        "Adaptive selection",
        value=True,
        help="Keep codes near each concept's own best score, rather than one flat "
        "threshold for every concept.",
    )
    threshold = st.slider("Threshold", 0.0, 1.0, 0.45, 0.01, disabled=adaptive)
    rel_margin = st.slider("Relative margin", 0.0, 0.5, 0.10, 0.01, disabled=not adaptive)

    if st.button("Run similarity", type="primary"):
        try:
            with st.spinner("Embedding and scoring…"):
                result = api.run_similarity(
                    ontology_id=ontology["id"],
                    level=level,
                    adaptive=adaptive,
                    threshold=threshold,
                    rel_margin=rel_margin,
                )
            st.session_state["run_id"] = result["run_id"]
            st.success(
                f"Run {result['run_id']}: {result['n_scores']} scores, "
                f"{result['n_linked']} proposed, "
                f"{result['n_manual_preserved']} manual link(s) preserved."
            )
        except ApiError as exc:
            st.error(exc.detail)

concepts = api.list_concepts(ontology["id"])
if not concepts:
    st.info("This ontology has no concepts yet.")
    st.stop()

run_id = st.session_state.get("run_id")
if run_id is None:
    st.info("Run similarity in the sidebar to see candidate codes.")

for concept in concepts:
    links = api.concept_links(concept["id"])
    manual_count = sum(1 for link in links if link["manual"])

    header = f"{concept['name']} — {len(links)} link(s)"
    if manual_count:
        header += f", {manual_count} curated"

    with st.expander(header):
        if links:
            st.markdown("**Current links**")
            for link in links:
                mark = "✋ curated" if link["manual"] else f"🤖 run {link['proposed_by_run']}"
                score = f"{link['score']:.3f}" if link["score"] is not None else "—"
                st.markdown(
                    f"`{link['code']['code']}` {link['code']['name'] or ''} · {score} · {mark}"
                )

        if run_id is None:
            continue

        st.markdown("**Candidates**")
        try:
            candidates = api.concept_candidates(concept["id"], run_id)
        except ApiError as exc:
            st.error(exc.detail)
            continue

        if not candidates:
            st.caption("No scored candidates for this concept in the latest run.")
            continue

        for candidate in candidates:
            code = candidate["code"]
            key = f"link_{concept['id']}_{code['id']}"
            checked = st.checkbox(
                f"`{code['code']}` {code['name'] or ''} · {candidate['score']:.3f}",
                value=candidate["linked"],
                key=key,
            )
            if checked != candidate["linked"]:
                try:
                    api.set_link(concept["id"], code["id"], linked=checked)
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)

        st.caption(
            "Ticking a candidate stores it as a curated link, so the next "
            "similarity run will not remove it."
        )
