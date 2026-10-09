"""Thin HTTP client for the API.

The UI has no database access at all. Everything it shows or changes goes through
here, which is why the Streamlit layer can be replaced without moving any logic
out of it.

`Api` is one object with a method per endpoint; each API area's methods live in
a module of their own, as a mixin over the shared transport in `base`.
"""

from __future__ import annotations

from hontology.apps.ui.client.base import ApiError
from hontology.apps.ui.client.evaluation import EvaluationCalls
from hontology.apps.ui.client.filtering import FilteringCalls
from hontology.apps.ui.client.judgement import JudgementCalls
from hontology.apps.ui.client.labels import LabelCalls
from hontology.apps.ui.client.ontologies import OntologyCalls
from hontology.apps.ui.client.retrieval import RetrievalCalls
from hontology.apps.ui.client.runs import RunCalls

__all__ = ["Api", "ApiError"]


class Api(
    OntologyCalls,
    FilteringCalls,
    RetrievalCalls,
    JudgementCalls,
    LabelCalls,
    RunCalls,
    EvaluationCalls,
):
    """The API client: every endpoint the UI calls."""
