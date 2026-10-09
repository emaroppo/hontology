"""Judgement: what the model says about a class, and how a prompt edit changes it.

Pick a class and a few articles, and the model is asked about each exactly as a
chosen run asks it: the same model, decoding settings and article text limit.
The prompt is shown in full and can be edited, both its system text and the
class's wording, and every article is then asked again with the edits beside the
original, against its label and what the run recorded.

Nothing a trial says is stored. Wording that works can be saved to the class,
which is then a new ontology version like any other edit.

Only per-pair prompts can be tried: one system text and one message per article
and class. Batched, hierarchical and extraction prompts are several calls whose
shape depends on earlier answers, so a run that used one is shown for what it
recorded, and the trial asks the per-pair way.
"""

from __future__ import annotations

from functools import partial

from hontology.apps.ui import panel, shared
from hontology.apps.ui.judgement.context import Context
from hontology.apps.ui.judgement.evaluation import show_evaluation
from hontology.apps.ui.judgement.trials import show_trials

api, ontologies = shared.page("Judgement", "⚖️")

# The ontology, run and truth chosen in Settings.
ontology = panel.ontology(api, ontologies)
params = panel.current(api)
runs = api.judgement_runs(ontology["id"])
run = next((r for r in runs if r["id"] == params.run_id), None)
templates = {t["prompt_id"]: t for t in api.judgement_templates()}
ctx = Context(api, ontology, params.annotator, run, templates)

shared.lazy_tabs(
    "judgement_tab",
    {"Try a prompt": partial(show_trials, ctx), "Evaluation": partial(show_evaluation, ctx)},
)
