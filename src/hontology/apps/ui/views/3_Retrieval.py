"""Retrieval: what a run's search ranks, and where its cutoff should fall.

Retrieval ranks an ontology's leaves against each article by embedding
similarity, and a cutoff decides which pairs go on to the judge. Laid out as
Judgement is:

**Try a cutoff** shows the ranking for one thing at a time, your own pasted
text first, or a corpus article, or the articles that rank one class highest,
as the run chosen in Settings ranks them. The cutoff sits in a collapsible
section, starting from that run's own. Corpus articles need no embedding: every
run stores its whole ranked pool, so a cutoff is a different line through a
ranking that already exists.

**Evaluation** scores a retrieval *version* rather than a run: the embedding
settings, the leaves' wording and the ranking code. A cutoff is a parameter, set
by hand or loaded from a run that used the version. Recall is computed live over
every labelled article; cost, the pairs sent to the judge, over the articles of
the run whose ranking the version is.
"""

from __future__ import annotations

from functools import partial

from hontology.apps.ui import panel, shared
from hontology.apps.ui.retrieval.context import Context
from hontology.apps.ui.retrieval.evaluation import show_evaluation
from hontology.apps.ui.retrieval.try_tab import show_try

api, ontologies = shared.page("Retrieval", "🔎")

# The ontology, run and truth chosen in Settings; the cutoff, in the page.
ontology = panel.ontology(api, ontologies)
params = panel.current(api)
runs = api.retrieval_runs(ontology["id"])
run = next((r for r in runs if r["id"] == params.run_id), None)
ctx = Context(api, ontology, params.annotator, run)

shared.lazy_tabs(
    "retrieval_tab",
    {"Try a cutoff": partial(show_try, ctx), "Evaluation": partial(show_evaluation, ctx)},
)
