"""OpenRouter provider: the host is pinned, recorded, and checked.

No network — HTTP is faked. Each refusal guards a way a run's recorded config
could describe something other than what actually served it.
"""

from __future__ import annotations

import httpx
import pytest

from hontology.pipeline.judge.providers import openrouter
from hontology.pipeline.judge.providers.base import GenerationConfig, ProviderError
from hontology.pipeline.judge.providers.openrouter import OpenRouterChatProvider
from hontology.pipeline.runs.config import ConfigError, normalize

ROUTING = {"provider": "DeepInfra", "quantizations": ["bf16", "fp8"]}
MODEL = "google/gemma-4-26b-a4b-it"


def reply(content="{}", provider="DeepInfra", total=100):
    return {
        "provider": provider,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": total - 10, "completion_tokens": 10, "total_tokens": total},
    }


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, *, json, headers, **kwargs):
        self.posts.append({"url": url, "json": json, "headers": headers})
        status, body = self.responses.pop(0)
        request = httpx.Request("POST", url)
        if isinstance(body, dict):
            return httpx.Response(status, json=body, request=request)
        return httpx.Response(status, text=body, request=request)

    def get(self, url, **kwargs):
        body = {"data": {"limit": 10.0, "limit_remaining": 7.5}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))


@pytest.fixture
def api(monkeypatch):
    def install(*responses):
        fake = FakeAPI(responses)
        monkeypatch.setattr(openrouter.httpx, "post", fake.post)
        monkeypatch.setattr(openrouter.httpx, "get", fake.get)
        return fake

    return install


def provider():
    return OpenRouterChatProvider("sk-test", ROUTING, sleep=lambda s: None)


def complete(**kwargs):
    defaults = dict(system="sys", prompt="p", config=GenerationConfig(), model=MODEL)
    return provider().complete(**{**defaults, **kwargs})


class TestRequest:
    def test_the_host_is_pinned_without_fallbacks(self, api):
        fake = api((200, reply()))
        complete()
        routing = fake.posts[0]["json"]["provider"]
        assert routing["order"] == ["DeepInfra"]
        assert routing["allow_fallbacks"] is False
        assert routing["quantizations"] == ["bf16", "fp8"]
        assert routing["data_collection"] == "deny"
        assert routing["require_parameters"] is True

    def test_thinking_is_switched_off_explicitly(self, api):
        fake = api((200, reply()))
        complete(want_reasoning=False)
        assert fake.posts[0]["json"]["reasoning"] == {"enabled": False}

    def test_json_mode_and_settings_are_sent(self, api):
        fake = api((200, reply()))
        complete(config=GenerationConfig(temperature=0.2, seed=3, max_output_tokens=40))
        body = fake.posts[0]["json"]
        assert body["response_format"] == {"type": "json_object"}
        assert (body["temperature"], body["seed"], body["max_tokens"]) == (0.2, 3, 40)

    def test_the_key_goes_in_the_header_only(self, api):
        fake = api((200, reply()))
        complete()
        assert fake.posts[0]["headers"] == {"Authorization": "Bearer sk-test"}
        assert "sk-test" not in str(fake.posts[0]["json"])

    def test_tokens_are_reported(self, api):
        api((200, reply(total=250)))
        result = complete()
        assert (result.input_tokens, result.output_tokens) == (240, 10)


class TestRefusals:
    def test_a_reply_from_another_host_is_refused(self, api):
        api((200, reply(provider="SomeOtherHost")))
        with pytest.raises(ProviderError, match="not the pinned 'DeepInfra'"):
            complete()

    def test_a_missing_key_or_unpinned_host_is_refused(self):
        with pytest.raises(ProviderError, match="API_KEY"):
            OpenRouterChatProvider(None, ROUTING)
        with pytest.raises(ProviderError, match="pin the host"):
            OpenRouterChatProvider("sk-test", {})

    def test_an_empty_answer_is_an_error(self, api):
        api((200, reply(content="  ")))
        with pytest.raises(ProviderError, match="empty response"):
            complete()


class TestRetries:
    def test_rate_limits_are_retried_then_succeed(self, api):
        fake = api((429, "slow down"), (503, "busy"), (200, reply(content='{"ok": 1}')))
        assert complete().text == '{"ok": 1}'
        assert len(fake.posts) == 3

    def test_retries_are_bounded(self, api):
        fake = api(*[(429, "slow down")] * openrouter.MAX_ATTEMPTS)
        with pytest.raises(ProviderError, match="429") as raised:
            complete()
        assert raised.value.retryable is True
        assert len(fake.posts) == openrouter.MAX_ATTEMPTS

    def test_a_refused_request_is_not_retried(self, api):
        """No credit, or a bad parameter: asking again changes nothing."""
        fake = api((402, "insufficient credits"))
        with pytest.raises(ProviderError, match="402"):
            complete()
        assert len(fake.posts) == 1


class TestHealth:
    def test_health_reports_the_pinned_host_and_the_keys_budget(self, api):
        api()
        assert provider().health() == (
            "openrouter: pinned to DeepInfra; $7.50 of $10.00 left on this key"
        )


class TestConfig:
    def test_an_unpinned_openrouter_config_is_refused(self):
        with pytest.raises(ConfigError, match="routing.provider is required"):
            normalize({"judge": {"provider": "openrouter", "model": MODEL}})

    def test_routing_is_only_for_openrouter(self):
        with pytest.raises(ConfigError, match="only to openrouter"):
            normalize({"judge": {"provider": "llamacpp", "routing": ROUTING}})

    def test_unknown_routing_keys_are_refused(self):
        with pytest.raises(ConfigError, match="unknown judge.routing keys"):
            normalize({"judge": {"provider": "openrouter", "routing": {**ROUTING, "x": 1}}})

    def test_configs_without_routing_keep_their_shape(self):
        """No routing key appears in a config that does not use it, so existing
        runs' stage keys are unchanged."""
        assert "routing" not in normalize({"judge": {"provider": "llamacpp"}})["judge"]
