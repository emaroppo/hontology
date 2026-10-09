"""Ontology editor: create ontologies, author concepts, import and export.

This is the page that has to work from a completely empty install, because it is
the first thing anyone sees.

Two kinds of ontology, edited differently:

- **Flat** (no ``subclass_of`` edges): a list of event classes, authored entirely
  here. The quickest way to try the pipeline on an event set made up on the spot.
- **Structured** (a class hierarchy): the structure, names and internal classes
  are authored in an OWL editor such as Protégé and arrive by import. This page
  shows the tree and edits only leaf wording, which is what the judge reads.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import panel, shared
from hontology.apps.ui.ontology.flat import show_add, show_flat_editor
from hontology.apps.ui.ontology.header import new_and_import, version_header
from hontology.apps.ui.ontology.health_export import show_export, show_health
from hontology.apps.ui.ontology.hierarchy import show_hierarchy

api, ontologies = shared.page("Ontology", "🧭", need_ontology=False)

# New and imported ontologies; the one shown is the one chosen in Settings.
new_and_import(api, ontologies)

params = panel.current(api)
selected = next((o for o in ontologies if o["id"] == params.ontology_id), None)
if selected is None:
    st.info(
        "**Nothing here yet.** Create an ontology above and add a few event classes, "
        "or import one from OWL or JSON."
    )
    st.stop()

ontology_id = selected["id"]
slug = selected["slug"]
tree = api.hierarchy(ontology_id)
classes: list[dict] = tree["classes"]
structured: bool = tree["structured"]
version_header(api, selected, classes, structured)

if structured:
    tabs = st.tabs(["Hierarchy", "Health", "Export"])
    with tabs[0]:
        show_hierarchy(api, ontology_id, classes)
    with tabs[1]:
        show_health(api, ontology_id)
    with tabs[2]:
        show_export(api, ontology_id, slug)
else:
    tabs = st.tabs(["Edit classes", "Add classes", "Health", "Export"])
    with tabs[0]:
        show_flat_editor(api, ontology_id, classes)
    with tabs[1]:
        show_add(api, ontology_id)
    with tabs[2]:
        show_health(api, ontology_id)
    with tabs[3]:
        show_export(api, ontology_id, slug)
