"""The UI's entry point: navigation in the sidebar, shared settings top right.

Run with ``streamlit run src/hontology/ui/app.py`` (``make ui``). Pages are
listed here in pipeline order; each is an ordinary script in ``views/``,
not ``pages/``: Streamlit lists a ``pages/`` folder on its own until this
script has run, and a click in that moment opened a page without the settings.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import Api

st.set_page_config(page_title="hontology", page_icon="🧭", layout="wide")

shared.settings(Api())

st.navigation(
    [
        st.Page("Home.py", title="Home", icon="🧭", default=True),
        st.Page("views/1_Ontology.py", title="Ontology", icon="📚"),
        st.Page("views/2_Filtering.py", title="Filtering", icon="🧹"),
        st.Page("views/3_Retrieval.py", title="Retrieval", icon="🔎"),
        st.Page("views/4_Judgement.py", title="Judgement", icon="⚖️"),
        st.Page("views/5_Runs.py", title="Runs", icon="⚙️"),
        st.Page("views/6_Labelling.py", title="Labelling", icon="🏷️"),
    ],
).run()
