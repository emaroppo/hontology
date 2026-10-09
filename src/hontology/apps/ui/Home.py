"""Landing page: the scored runs, end to end.

The app starts with nothing in it, the premise being that the ontology is yours,
so with no ontology this page makes the empty state actionable. Otherwise it
opens on the leaderboard: every judged run scored on the same labelled sample,
the way the arms comparison scores it.

- **Leaderboard**: one row per run, with its stage versions, how much of the
  sample it covered, end-to-end precision, recall and F1, and its cost.
- **Comparison**: arms against a baseline, paired, as `eval arms` reports them.
- **Single run**: one run end to end, with a couple of numbers per stage; each
  stage's own evaluation lives on its page.
"""

from __future__ import annotations

from functools import partial

import streamlit as st

from hontology.apps.ui import panel, shared
from hontology.apps.ui.client import Api, ApiError
from hontology.apps.ui.home.comparison import show_comparison
from hontology.apps.ui.home.data import Context, scored_runs
from hontology.apps.ui.home.leaderboard import show_leaderboard
from hontology.apps.ui.home.single_run import show_single

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


def how_it_fits() -> None:
    with st.expander("How the pieces fit together"):
        st.markdown(
            """
            1. **Ontology**: describe what you care about: classes, their
               definitions, and the criteria that draw their boundaries.
            2. **Filtering**: link classes to the codes the feed tags articles
               with, so only articles worth reading are downloaded.
            3. **Retrieval**: rank classes against each article, and cut.
            4. **Judgement**: ask a model whether each article reports each class.
            5. **Labelling**: build ground truth, a sample at a time.
            6. **Here**: score runs end to end; each stage page scores its own stage.
            """
        )


if not ontologies:
    st.info("**No ontologies yet.** This install is empty by design.")
    st.markdown("Head to the **Ontology** page to create one, or import one from OWL or JSON.")
    how_it_fits()
    st.stop()
params = panel.current(api)
ontology = next((o for o in ontologies if o["id"] == params.ontology_id), ontologies[0])

include_partial = bool(st.session_state.get("home_partial", False))
try:
    board = scored_runs(
        ontology["id"],
        params.sample,
        params.annotator,
        include_partial,
        st.session_state.get("home_recompute", 0),
    )
except ApiError as exc:
    st.error(exc.detail)
    how_it_fits()
    st.stop()

ctx = Context(api, params, board, st.session_state.setdefault("home_calendar", {}))

shared.lazy_tabs(
    "home_tab",
    {
        "Leaderboard": partial(show_leaderboard, ctx),
        "Comparison": partial(show_comparison, ctx),
        "Single run": partial(show_single, ctx),
    },
)

st.divider()
how_it_fits()
