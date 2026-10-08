"""The truth a page scores against: human labels, or a machine annotation set.

One picker for every page, so the choice reads the same everywhere and a score
always says which kind of truth it was computed against.
"""

from __future__ import annotations

from typing import Any

from hontology.ui.client import Api


def pick(api: Api, ontology_id: int, where: Any, *, key: str) -> str | None:
    """A radio in *where* (the sidebar, a column); returns None for human labels,
    else the chosen machine annotation set's name."""
    truths = api.truths(ontology_id)
    human = truths["human"]
    options: dict[str, str | None] = {f"Human labels ({human['articles']} articles)": None}
    descriptions: dict[str, str | None] = {}
    for found in truths["machine"]:
        label = f"Machine: {found['name']} ({found['articles']} articles)"
        options[label] = found["name"]
        descriptions[found["name"]] = found["description"]
    chosen = options[
        where.radio(
            "Truth",
            list(options),
            key=key,
            help="Human labels are the label bank, trusted and current only. A machine "
            "annotation set is an annotator's own labels: a score against it measures "
            "agreement with that annotator, not correctness.",
        )
    ]
    if chosen is not None and descriptions.get(chosen):
        where.caption(descriptions[chosen])
    return chosen
