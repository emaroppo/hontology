"""Labelling: where ground truth gets made, in two ways.

**Sample** labels a frozen random sample a whole document at a time. A labelled
sample is what arms are scored against, so it is labelled blind: this tab shows
the article and the leaves, never a run's verdicts, and not the calendar stratum
the document was drawn from, which names the event it was expected to report.
Documents come in the sample's frozen order, so whatever prefix is done is itself
a random sample. Saving answers every leaf at once: the ticked ones apply, the
rest do not.

**Queue** labels single pairs, ranked by how much a label would teach, and
adjudicates machine proposals. It has to answer two questions fast: *why is this
pair in front of me*, and *what did the models say*. Both are shown, because a
queue that hides its reasoning trains you to rubber stamp it — and a
rubber-stamped machine proposal is exactly the label that poisons the metrics it
is supposed to validate.

Sample comes first so the page opens blind: the queue shows run verdicts, and
only the open tab is run at all.
"""

from __future__ import annotations

from functools import partial

from hontology.apps.ui import shared
from hontology.apps.ui.labelling.queue import queue_tab
from hontology.apps.ui.labelling.sample import sample_tab

api, ontologies = shared.page("Labelling", "🏷️")

shared.lazy_tabs(
    "labelling_tab",
    {
        "Sample (whole documents, blind)": partial(sample_tab, api, ontologies),
        "Queue (pairs and review)": partial(queue_tab, api, ontologies),
    },
)
