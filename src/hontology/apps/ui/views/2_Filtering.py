"""Filtering: which feed articles are worth downloading for an ontology.

The GDELT feeds tag every article before anything is fetched: the event export
with CAMEO event codes, the knowledge graph with GKG themes. Linking a class to
codes lets the optional pre-download filter fetch only articles whose codes link
to some class. That gate bounds recall for everything downstream, so this page
puts the three things needed to set it well side by side: the links, what they
would keep, and what each code has been worth once articles were labelled.

Every link is ticked by hand. A similarity run only ranks codes against each
class, so its proposals appear as candidates; proposing links automatically let
far too much through.
"""

from __future__ import annotations

from functools import partial

import streamlit as st

from hontology.apps.ui import panel, shared
from hontology.apps.ui.filtering import context
from hontology.apps.ui.filtering.codebooks import show_codebooks
from hontology.apps.ui.filtering.evaluation import show_evaluation
from hontology.apps.ui.filtering.links import show_links
from hontology.apps.ui.filtering.preview import show_preview

api, ontologies = shared.page(
    "Filtering",
    "🧹",
    caption="Which feed articles are worth downloading. Classes link to the codes the "
    "feeds tag articles with (CAMEO event codes, GKG themes), and the optional "
    "pre-download filter fetches only articles whose codes link to some class. "
    "Semantic retrieval does not use the links, so an ontology can do without any.",
)
ontology = panel.ontology(api, ontologies)
ctx = context.load(api, ontology["id"])
classes, links = ctx.classes, ctx.links

if not classes:
    st.info("This ontology has no classes yet. Add some on the **Ontology** page.")
    st.stop()

linked_by_system: dict[str, int] = {}
for rows in links.values():
    for link in rows:
        system = link["code"]["system"]
        linked_by_system[system] = linked_by_system.get(system, 0) + 1

head = st.columns(4)
head[0].metric("Classes with links", f"{len(links)} of {len(classes)}")
head[1].metric("CAMEO links", linked_by_system.get("cameo", 0))
head[2].metric("Theme links", linked_by_system.get("gkg-themes", 0))
head[3].metric("Filter", "usable" if links else "no links yet")

message = st.session_state.pop("filtering_message", None)
if message:
    st.success(message)

shared.lazy_tabs(
    "filtering_tab",
    {
        "Links": partial(show_links, ctx),
        "What it keeps": partial(show_preview, ctx),
        "Evaluation": partial(show_evaluation, ctx),
        "Codebooks": partial(show_codebooks, ctx),
    },
)
