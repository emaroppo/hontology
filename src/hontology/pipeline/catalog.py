"""Which models each provider serves, for a form to offer.

A run needs two models: an embedding model for retrieval and a decoder for the
judge (and for routing and extraction). They are picked separately and may come
from different providers: embeddings from a local Ollama, answers from a
llama.cpp server elsewhere on the network. Each provider is asked at the host
the run would use for that kind of model.

Ollama does not say which of its models embed, so every one is listed. A
llama.cpp model is listed by its file name's stem, the portable name a config
stores. A provider that cannot be reached raises a `ProviderError`; a form then
falls back to typing the name.
"""

from __future__ import annotations

from hontology.config import get_settings
from hontology.pipeline.judge.providers.base import ProviderError
from hontology.pipeline.judge.providers.http import get_json
from hontology.pipeline.judge.providers.llamacpp import model_stem, served_models

EMBEDDING, DECODER = "embedding", "decoder"
PROVIDERS = {
    EMBEDDING: ["ollama", "llamacpp"],
    DECODER: ["ollama", "llamacpp", "openrouter"],
}


def list_models(provider: str, kind: str) -> list[str]:
    """The models *provider* serves for *kind* of use, sorted by name."""
    if provider not in PROVIDERS.get(kind, []):
        raise ValueError(f"{provider!r} does not serve {kind} models here")
    settings = get_settings()
    if provider == "ollama":
        host = settings.ollama_host.rstrip("/")
        tags = get_json(f"{host}/api/tags", unreachable=f"Ollama unreachable at {host}")
        names = [m["name"] for m in tags.get("models", [])]
    elif provider == "llamacpp":
        host = (
            settings.llamacpp_embed_host or settings.llamacpp_host
            if kind == EMBEDDING
            else settings.llamacpp_host
        ).rstrip("/")
        names = [
            model_stem(m.get("id", "")) for m in served_models(host, settings.llamacpp_api_key)
        ]
    else:
        base = settings.openrouter_base_url.rstrip("/")
        catalogue = get_json(
            f"{base}/models", unreachable=f"OpenRouter unreachable at {base}", timeout=15.0
        )
        names = [m["id"] for m in catalogue.get("data", [])]
    return sorted({n for n in names if n})


__all__ = ["DECODER", "EMBEDDING", "PROVIDERS", "ProviderError", "list_models"]
