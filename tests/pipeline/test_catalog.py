"""Which models a provider serves, asked at the host each kind of model uses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hontology.apps.api.main import app
from hontology.pipeline import catalog
from hontology.pipeline.judge.providers.base import ProviderError


@pytest.fixture
def hosts(monkeypatch):
    settings = SimpleNamespace(
        ollama_host="http://ollama.test",
        llamacpp_host="http://chat.test",
        llamacpp_embed_host="http://embed.test",
        llamacpp_api_key=None,
        openrouter_base_url="https://openrouter.test/api/v1",
    )
    monkeypatch.setattr(catalog, "get_settings", lambda: settings)
    asked: list[str] = []

    def get_json(url, **kwargs):
        asked.append(url)
        if url.startswith("http://ollama.test"):
            return {"models": [{"name": "nomic-embed-text:latest"}, {"name": "gemma3:4b"}]}
        raise ProviderError("unreachable")

    def served_models(host, api_key=None):
        asked.append(host)
        return [{"id": f"/models/{host.split('//')[1].split('.')[0]}-model.gguf"}]

    monkeypatch.setattr(catalog, "get_json", get_json)
    monkeypatch.setattr(catalog, "served_models", served_models)
    return asked


def test_each_kind_of_model_is_listed_from_its_own_host(hosts):
    assert catalog.list_models("ollama", catalog.EMBEDDING) == [
        "gemma3:4b",
        "nomic-embed-text:latest",
    ]
    # llama.cpp models are listed by their portable stem, at the host for the kind.
    assert catalog.list_models("llamacpp", catalog.EMBEDDING) == ["embed-model"]
    assert catalog.list_models("llamacpp", catalog.DECODER) == ["chat-model"]
    assert hosts[1:] == ["http://embed.test", "http://chat.test"]


def test_a_provider_that_serves_no_such_kind_is_refused(hosts):
    with pytest.raises(ValueError):
        catalog.list_models("openrouter", catalog.EMBEDDING)


def test_the_api_says_why_a_provider_could_not_list(hosts):
    client = TestClient(app)
    ok = client.get("/runs/models", params={"provider": "ollama", "kind": "decoder"})
    assert ok.status_code == 200 and "gemma3:4b" in ok.json()
    down = client.get("/runs/models", params={"provider": "openrouter", "kind": "decoder"})
    assert down.status_code == 503 and "unreachable" in down.json()["detail"]
