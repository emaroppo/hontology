"""One labelled model widget: a provider, and a model that provider serves.

Anything needing both an embedding model and a decoder shows two of these,
labelled, each picked on its own: embeddings may come from a local Ollama and
answers from a llama.cpp server elsewhere. The models offered are the ones the
provider serves now; one it cannot be asked about is typed instead, and a
value already chosen stays on offer even when the provider no longer lists it.
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import Api, ApiError

EMBEDDING, DECODER = "embedding", "decoder"
LABELS = {EMBEDDING: "Embedding model", DECODER: "Decoder model"}
HELPS = {
    EMBEDDING: "Ranks the classes against an article: retrieval's first step.",
    DECODER: "Answers the judge's questions, and routes or extracts events where "
    "the prompt does.",
}


def _same(a: str, b: str) -> bool:
    return a.removesuffix(":latest") == b.removesuffix(":latest")


@st.cache_data(ttl=60, show_spinner=False)
def _served(_api: Api, provider: str, kind: str) -> tuple[list[str], str | None]:
    try:
        return _api.provider_models(provider, kind), None
    except ApiError as exc:
        return [], exc.detail


def model_picker(
    api: Api, kind: str, providers: list[str], provider_key: str, model_key: str
) -> tuple[str, str]:
    """Draw the widget; returns the provider and model chosen.

    *provider_key* and *model_key* are the session-state keys holding the two
    values, so a form can seed them and read them back.
    """
    with st.container(border=True):
        st.markdown(f"**{LABELS[kind]}**", help=HELPS[kind])
        cols = st.columns([1, 2])
        provider = cols[0].selectbox("Provider", providers, key=provider_key)
        served, error = _served(api, provider, kind)
        current = st.session_state.get(model_key) or ""
        if error or not served:
            cols[1].text_input(
                "Model",
                key=model_key,
                help=f"{provider} could not list its models ({error}); type the name."
                if error
                else f"{provider} serves no models now; type the name.",
            )
        else:
            # Ollama serves "name" as "name:latest"; the stored spelling is kept,
            # since a respelled model would change the run's key.
            served = [current if _same(m, current) else m for m in served]
            options = served if current in served or not current else [current, *served]
            if current not in options:
                st.session_state[model_key] = options[0]
            cols[1].selectbox(
                "Model",
                options,
                key=model_key,
                format_func=lambda m: m if m in served else f"{m} (not served now)",
            )
    return provider, st.session_state.get(model_key) or ""


@st.cache_data(ttl=300, show_spinner=False)
def _providers(_api: Api) -> dict[str, list[str]]:
    choices = _api.run_options()["choices"]
    return {EMBEDDING: choices["embed_providers"], DECODER: choices["judge_providers"]}


def model_widgets(
    api: Api, run_id: int, own: dict[str, list[str]], kinds: list[str], prefix: str
) -> dict[str, tuple[str, str]]:
    """The labelled widgets for *kinds*, side by side, each seeded from *own*,
    the run's (provider, model) per kind; returns what is picked per kind.

    Choosing another run starts again from its models; a widget hidden for a
    while comes back with the run's model rather than an empty one.
    """
    marker = f"{prefix}:models_for"
    fresh = st.session_state.get(marker) != run_id
    st.session_state[marker] = run_id
    for kind, (provider, model) in own.items():
        for name, seed in (("provider", provider), ("model", model)):
            if fresh or f"{prefix}:{kind}:{name}" not in st.session_state:
                st.session_state[f"{prefix}:{kind}:{name}"] = seed
    try:
        providers = _providers(api)
    except ApiError as exc:
        st.error(exc.detail)
        st.stop()
    picked = {}
    for column, kind in zip(st.columns(len(kinds)), kinds, strict=True):
        with column:
            picked[kind] = model_picker(
                api,
                kind,
                providers[kind],
                f"{prefix}:{kind}:provider",
                f"{prefix}:{kind}:model",
            )
    return picked


def is_own(picked: tuple[str, str], own: list[str]) -> bool:
    """Whether *picked* is the run's own (provider, model)."""
    return picked[0] == own[0] and _same(picked[1], own[1])
