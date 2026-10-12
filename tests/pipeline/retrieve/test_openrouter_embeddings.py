"""OpenRouter embeddings: the host is pinned and checked, as for the hosted judge.

No network: HTTP is faked with the judge tests' fake API.
"""

from __future__ import annotations

import pytest

from hontology.pipeline.judge.providers.base import ProviderError
from hontology.pipeline.retrieve import embed
from hontology.pipeline.retrieve.embedders import OpenRouterEmbeddingProvider
from hontology.pipeline.runs.config import ConfigError, normalize
from tests.pipeline.judge.test_openrouter import FakeAPI, api  # noqa: F401

ROUTING = {"provider": "DeepInfra"}
MODEL = "qwen/qwen3-embedding-8b"


def vectors(*rows, provider="DeepInfra"):
    # Out of order on purpose: the index, not the order, places each vector.
    data = [{"index": i, "embedding": row} for i, row in enumerate(rows)][::-1]
    return {"provider": provider, "data": data}


def provider():
    return OpenRouterEmbeddingProvider("key", ROUTING, sleep=lambda s: None)


def test_texts_are_embedded_from_the_pinned_host_in_order(api):  # noqa: F811
    fake = api((200, vectors([1.0, 0.0], [0.0, 1.0])))
    assert provider().embed(["a", "b"], model=MODEL) == [[1.0, 0.0], [0.0, 1.0]]
    sent = fake.posts[0]
    assert sent["url"].endswith("/embeddings")
    assert sent["json"]["provider"]["order"] == ["DeepInfra"]
    assert sent["json"]["provider"]["allow_fallbacks"] is False


def test_vectors_from_another_host_are_refused(api):  # noqa: F811
    api((200, vectors([1.0, 0.0], provider="Elsewhere")))
    with pytest.raises(ProviderError, match="not the pinned"):
        provider().embed(["a"], model=MODEL)


def test_a_rate_limit_is_retried(api):  # noqa: F811
    fake = api((429, "slow down"), (200, vectors([1.0, 0.0])))
    assert provider().embed(["a"], model=MODEL) == [[1.0, 0.0]]
    assert len(fake.posts) == 2


def test_the_pinned_host_is_part_of_the_cache_key():
    assert provider().name == "openrouter@deepinfra"
    assert embed.provider_key("openrouter", ROUTING) == "openrouter@deepinfra"
    assert embed.provider_key("ollama") == "ollama"


def test_a_host_must_be_pinned():
    with pytest.raises(ProviderError, match="pin the host"):
        OpenRouterEmbeddingProvider("key", None)
    with pytest.raises(ConfigError, match="candidates.embed_routing.provider is required"):
        normalize({"candidates": {"embed_provider": "openrouter"}})
    with pytest.raises(ConfigError, match="applies only to openrouter"):
        normalize({"candidates": {"embed_routing": ROUTING}})


def test_configs_without_a_hosted_embedder_are_unchanged():
    assert "embed_routing" not in normalize({})["candidates"]
    hosted = normalize(
        {"candidates": {"embed_provider": "openrouter", "embed_routing": ROUTING}}
    )
    assert hosted["candidates"]["embed_routing"] == ROUTING
