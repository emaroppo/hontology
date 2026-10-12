"""One labelled model widget: a provider, and a model that provider serves.

Anything needing both an embedding model and a decoder shows two of these,
labelled, each picked on its own: embeddings may come from a local Ollama and
answers from a llama.cpp server elsewhere. The models offered are the ones the
provider serves now; one it cannot be asked about is typed instead, and a
value already chosen stays on offer even when the provider no longer lists it.

A hosted provider (OpenRouter) also needs the host pinned, since hosts of one
model run it at different precisions; the widget then asks for it, and what is
picked is (provider, model, host) rather than (provider, model).
"""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import Api, ApiError

EMBEDDING, DECODER = "embedding", "decoder"
HOSTED = "openrouter"
LABELS = {EMBEDDING: "Embedding model", DECODER: "Decoder model"}
HELPS = {
    EMBEDDING: "Ranks the classes against an article: retrieval's first step.",
    DECODER: "Answers the judge's questions, and routes or extracts events where "
    "the prompt does.",
}

Picked = tuple[str, ...]
NAMES = ("provider", "model", "host")


def _same(a: str, b: str) -> bool:
    return a.removesuffix(":latest") == b.removesuffix(":latest")


@st.cache_data(ttl=60, show_spinner=False)
def _served(_api: Api, provider: str, kind: str) -> tuple[list[str], str | None]:
    try:
        return _api.provider_models(provider, kind), None
    except ApiError as exc:
        return [], exc.detail


def model_picker(
    api: Api, kind: str, providers: list[str], provider_key: str, model_key: str, host_key: str
) -> Picked:
    """Draw the widget; returns (provider, model), or (provider, model, host)
    for a hosted provider.

    The keys are the session-state keys holding the values, so a form can seed
    them and read them back.
    """
    with st.container(border=True):
        st.markdown(f"**{LABELS[kind]}**", help=HELPS[kind])
        provider = st.session_state.get(provider_key)
        cols = st.columns([3, 4, 2] if provider == HOSTED else [1, 2])
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
        model = st.session_state.get(model_key) or ""
        if provider != HOSTED:
            return provider, model
        # The provider is read before its widget is drawn, to make room for the
        # host; Streamlit has already applied a change of provider by then.
        (cols[2] if len(cols) == 3 else st).text_input(
            "Host",
            key=host_key,
            placeholder="required",
            help="The company serving the model, pinned so every call in a run is "
            "served the same way; a reply from another host is refused.",
        )
    return provider, model, (st.session_state.get(host_key) or "").strip()


@st.cache_data(ttl=300, show_spinner=False)
def _providers(_api: Api) -> dict[str, list[str]]:
    choices = _api.run_options()["choices"]
    return {EMBEDDING: choices["embed_providers"], DECODER: choices["judge_providers"]}


def model_widgets(
    api: Api, run_id: int, own: dict[str, list[str]], kinds: list[str], prefix: str
) -> dict[str, Picked]:
    """The labelled widgets for *kinds*, side by side, each seeded from *own*,
    the run's (provider, model[, host]) per kind; returns what is picked per kind.

    Choosing another run starts again from its models; a widget hidden for a
    while comes back with the run's model rather than an empty one.
    """
    marker = f"{prefix}:models_for"
    fresh = st.session_state.get(marker) != run_id
    st.session_state[marker] = run_id
    for kind, seeds in own.items():
        for name, seed in zip(("provider", "model", "host"), [*seeds, "", ""][:3], strict=True):
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
                api, kind, providers[kind], *(f"{prefix}:{kind}:{n}" for n in NAMES)
            )
    return picked


def is_own(picked: Picked, own: list[str]) -> bool:
    """Whether *picked* is the run's own (provider, model[, host])."""
    return (
        len(picked) == len(own)
        and picked[0] == own[0]
        and _same(picked[1], own[1])
        and picked[2:] == tuple(own[2:])
    )
