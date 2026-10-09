"""What every Filtering tab reads: the ontology's classes, their links, the
loaded codebooks and this session's latest similarity run."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from hontology.apps.ui.client import Api

SYSTEMS = {"cameo": "CAMEO event codes", "gkg-themes": "GKG themes"}


@dataclass(frozen=True)
class Context:
    api: Api
    ontology_id: int
    classes: list[dict]
    # Links by class id.
    links: dict[int, list[dict]]
    # Loaded code systems by slug.
    systems: dict[str, dict]
    # The latest similarity run this session, whose scores are the candidates.
    similarity_run: dict | None
    # What the links are now, so a preview or report computed before a tick can
    # say it is out of date.
    fingerprint: tuple


def load(api: Api, ontology_id: int) -> Context:
    tree = api.hierarchy(ontology_id)
    links = api.ontology_links(ontology_id)
    systems = {s["slug"]: s for s in api.code_systems()}
    return Context(
        api=api,
        ontology_id=ontology_id,
        classes=tree["classes"],
        links=links,
        systems=systems,
        similarity_run=st.session_state.get("similarity_runs", {}).get(ontology_id),
        fingerprint=tuple(
            sorted((c, link["code"]["id"]) for c, rows in links.items() for link in rows)
        ),
    )
