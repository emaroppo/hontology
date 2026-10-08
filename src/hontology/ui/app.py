"""The UI's entry point: navigation in the sidebar, shared settings top right.

Run with ``streamlit run src/hontology/ui/app.py`` (``make ui``). Pages are
listed here in pipeline order; each is an ordinary script in ``pages/``.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import shared
from hontology.ui.client import Api

st.set_page_config(page_title="hontology", page_icon="🧭", layout="wide")

shared.settings(Api())

st.navigation(
    [
        st.Page("Home.py", title="Home", icon="🧭", default=True),
        st.Page("pages/1_Ontology.py", title="Ontology", icon="📚"),
        st.Page("pages/2_Filtering.py", title="Filtering", icon="🧹"),
        st.Page("pages/3_Retrieval.py", title="Retrieval", icon="🔎"),
        st.Page("pages/4_Judgement.py", title="Judgement", icon="⚖️"),
        st.Page("pages/5_Labelling.py", title="Labelling", icon="🏷️"),
        st.Page("pages/6_Runs.py", title="Runs", icon="⚙️"),
    ],
).run()
