"""llama.cpp provider: the request it sends and the server facts it refuses.

No server needed — HTTP is faked. Each refusal here guards a way the run's
recorded config could describe something other than what actually ran.
"""

from __future__ import annotations

import httpx
import pytest

from hontology.pipeline.judge.providers.base import GenerationConfig, ProviderError
from hontology.pipeline.judge.providers.llamacpp import LlamaCppChatProvider, model_stem
from hontology.pipeline.retrieve import embed

HOST = "http://llama:8080"
MODEL_PATH = "/models/gemma-4-26B-A4B-it-Q8_0.gguf"
MODELS = {"data": [{"id": MODEL_PATH, "aliases": ["gemma"]}]}
PROPS = {"default_generation_settings": {"n_ctx": 16384}}


def _reply(content="{}", reasoning=None, finish="stop", total=100):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": total - 10, "completion_tokens": 10, "total_tokens": total},
    }


class FakeServer:
    def __init__(self, reply=None, models=MODELS, props=PROPS, embed_status=200):
        self.reply = reply or _reply()
        self.models = models
        self.props = props
        self.embed_status = embed_status
        self.posts: list[dict] = []

    def get(self, url, **kwargs):
        body = self.models if url.endswith("/v1/models") else self.props
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    def post(self, url, *, json, **kwargs):
        self.posts.append(json)
        request = httpx.Request("POST", url)
        if url.endswith("/v1/embeddings"):
            if self.embed_status != 200:
                return httpx.Response(self.embed_status, text="no embeddings", request=request)
            # Deliberately out of order: the provider must sort by index.
            indices = reversed(range(len(json["input"])))
            data = [{"index": i, "embedding": [float(i)]} for i in indices]
            return httpx.Response(200, json={"data": data}, request=request)
        return httpx.Response(200, json=self.reply, request=request)


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(httpx, "get", fake.get)
    monkeypatch.setattr(httpx, "post", fake.post)
    return fake


def _complete(**kwargs):
    defaults = dict(system="sys", prompt="p", config=GenerationConfig(), model="gemma")
    return LlamaCppChatProvider(HOST).complete(**{**defaults, **kwargs})


class TestModelResolution:
    def test_stem_drops_the_path_and_extension(self):
        assert model_stem(MODEL_PATH) == "gemma-4-26B-A4B-it-Q8_0"
        assert model_stem("gemma") == "gemma"

    @pytest.mark.parametrize("name", [MODEL_PATH, "gemma-4-26B-A4B-it-Q8_0", "gemma"])
    def test_id_stem_or_alias_resolves_to_the_server_id(self, server, name):
        _complete(model=name)
        assert server.posts[0]["model"] == MODEL_PATH

    def test_a_model_the_server_is_not_holding_is_refused(self, server):
        """The server would answer with its own model and the run would be
        recorded under the wrong name."""
        with pytest.raises(ProviderError, match="not serving 'qwen3'"):
            _complete(model="qwen3")
        assert server.posts == []


class TestRequest:
    def test_thinking_is_switched_off_explicitly(self, server):
        """Left unset, a reasoning model's template thinks by default."""
        _complete(want_reasoning=False)
        assert server.posts[0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_json_mode_survives_reasoning(self, server):
        _complete(want_reasoning=True, want_json=True)
        payload = server.posts[0]
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert payload["response_format"] == {"type": "json_object"}

    def test_generation_settings_are_sent(self, server):
        _complete(config=GenerationConfig(temperature=0.3, seed=7, max_output_tokens=50))
        payload = server.posts[0]
        assert (payload["temperature"], payload["seed"], payload["max_tokens"]) == (0.3, 7, 50)
        assert payload["messages"][0] == {"role": "system", "content": "sys"}

    def test_unset_limits_are_left_to_the_server(self, server):
        _complete()
        assert "seed" not in server.posts[0] and "max_tokens" not in server.posts[0]


class TestResponse:
    def test_answer_reasoning_and_usage_are_carried(self, server):
        server.reply = _reply('{"matched": true}', reasoning="thought", total=120)
        completion = _complete()
        assert completion.text == '{"matched": true}'
        assert completion.reasoning == "thought"
        assert (completion.input_tokens, completion.output_tokens) == (110, 10)
        assert completion.stop_reason == "stop"

    def test_empty_answer_is_an_error_naming_the_finish_reason(self, server):
        """A token budget spent mid-thought leaves reasoning but no answer."""
        server.reply = _reply("", reasoning="*   ", finish="length")
        with pytest.raises(ProviderError, match="finish_reason=length"):
            _complete()


class TestContextWindow:
    def test_a_server_with_less_context_than_configured_is_refused(self, server):
        with pytest.raises(ProviderError, match="16384 tokens of context"):
            _complete(config=GenerationConfig(context_window=32768))
        assert server.posts == []

    def test_a_call_over_the_configured_window_is_an_error(self, server):
        """The server had room, but the run claims a smaller window."""
        server.reply = _reply(total=9000)
        with pytest.raises(ProviderError, match="9000 tokens"):
            _complete(config=GenerationConfig(context_window=8192))


class TestEmbeddings:
    def test_vectors_come_back_in_input_order(self, server):
        provider = embed.LlamaCppEmbeddingProvider(HOST, batch_size=2)
        vectors = provider.embed(["a", "b", "c"], model="gemma")
        assert vectors == [[0.0], [1.0], [0.0]]
        assert [p["input"] for p in server.posts] == [["a", "b"], ["c"]]

    def test_a_server_without_embeddings_is_not_retried(self, server):
        server.embed_status = 501
        with pytest.raises(ProviderError) as info:
            embed.LlamaCppEmbeddingProvider(HOST).embed(["a"], model="gemma")
        assert not info.value.retryable


class TestPrefixFamilies:
    @pytest.mark.parametrize(
        "model",
        ["nomic-embed-text", "nomic-embed-text:latest", "/m/nomic-embed-text-v1.5.Q8_0.gguf"],
    )
    def test_every_name_for_the_same_weights_gets_the_prefix(self, model):
        """An exact-name miss silently embeds without the trained prefix."""
        assert embed.query_prefix(model) == "search_query: "
        assert embed.document_prefix(model) == "search_document: "
