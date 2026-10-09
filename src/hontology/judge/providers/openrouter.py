"""OpenRouter-backed completion provider, over its OpenAI-compatible API.

OpenRouter forwards each request to one of several companies hosting the same
model, and they do not all run it the same way: precisions differ (full,
FP8, sometimes lower), and so can answers. Left to its default routing, two calls
in one run could be served by different setups, which is uncontrolled variation
in what should be a controlled comparison. So:

**1. The host is pinned, and part of the run.** A run's ``judge.routing`` names
the provider to use, the precisions it may run at, and whether fallbacks are
allowed (they are not, by default). It is behavior, so it is hashed with the
judge config; a config for this provider without a pinned host is refused when
it is normalized.

**2. A reply from anyone else is refused.** The response names the host that
served it; a reply from a host other than the pinned one is an error rather
than a verdict recorded under the wrong setup.

**3. Thinking is set explicitly**, both ways, as for llama.cpp: a reasoning
model left to its default may think, which costs tokens and makes ``think:
false`` untrue.

**4. Rate limits and server errors are retried** with backoff, honoring
``Retry-After``: a hosted API sheds load in a way a local server never does.
A refused request (bad parameters, no credit) is not retried.

The API key is infrastructure, read from ``HONTOLOGY_OPENROUTER_API_KEY`` and
never part of a config or a log line.
"""

from __future__ import annotations

import time

import httpx

from hontology.judge.providers.base import (
    ChatProvider,
    Completion,
    GenerationConfig,
    ProviderError,
)
from hontology.judge.providers.http import chat_payload, get_json, read_chat

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


def routing_payload(routing: dict) -> dict:
    """OpenRouter's ``provider`` request field, from a run's ``judge.routing``."""
    payload: dict = {
        "order": [routing["provider"]],
        "allow_fallbacks": bool(routing.get("allow_fallbacks", False)),
        # Only hosts that honor every parameter sent, JSON mode included.
        "require_parameters": True,
        "data_collection": routing.get("data_collection", "deny"),
    }
    if routing.get("quantizations"):
        payload["quantizations"] = list(routing["quantizations"])
    return payload


def _retry_after(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        try:
            return min(60.0, float(response.headers.get("retry-after", "")))
        except ValueError:
            pass
    return min(60.0, 2.0**attempt)


class OpenRouterChatProvider(ChatProvider):
    name = "openrouter"

    def __init__(
        self,
        api_key: str | None,
        routing: dict | None,
        base_url: str = DEFAULT_BASE_URL,
        sleep=time.sleep,
    ) -> None:
        if not api_key:
            raise ProviderError("openrouter needs HONTOLOGY_OPENROUTER_API_KEY")
        if not routing or not routing.get("provider"):
            raise ProviderError("openrouter needs judge.routing.provider: pin the host")
        self.api_key = api_key
        self.routing = routing
        self.base_url = base_url.rstrip("/")
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        config: GenerationConfig,
        want_json: bool = True,
        want_reasoning: bool = False,
        model: str = "",
    ) -> Completion:
        if not model:
            raise ProviderError("openrouter provider requires an explicit model")
        payload = chat_payload(model, system, prompt, config, want_json) | {
            "provider": routing_payload(self.routing),
            # See the module docstring: sent either way, never left to the default.
            "reasoning": {"enabled": want_reasoning},
        }
        started = time.monotonic()
        body = self._post(payload, config.timeout_s)

        served_by = body.get("provider")
        if served_by and served_by.lower() != str(self.routing["provider"]).lower():
            raise ProviderError(
                f"openrouter served {model!r} from {served_by!r}, not the pinned "
                f"{self.routing['provider']!r}"
            )
        return read_chat(
            body,
            label="openrouter",
            model=model,
            config=config,
            want_json=want_json,
            want_reasoning=want_reasoning,
            reasoning_key="reasoning",
            started=started,
        )

    def _post(self, payload: dict, timeout_s: float) -> dict:
        last: ProviderError | None = None
        for attempt in range(MAX_ATTEMPTS):
            response: httpx.Response | None = None
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout_s,
                )
                if response.status_code in RETRY_STATUS:
                    last = ProviderError(
                        f"openrouter returned {response.status_code}: {response.text[:300]}",
                        retryable=True,
                    )
                else:
                    response.raise_for_status()
                    body = response.json()
                    # Errors can also come back inside a 200, from the host.
                    if body.get("error"):
                        error = body["error"]
                        code = error.get("code") if isinstance(error, dict) else None
                        raise ProviderError(
                            f"openrouter error: {str(error)[:300]}",
                            retryable=code in RETRY_STATUS,
                        )
                    return body
            except httpx.TimeoutException:
                last = ProviderError(f"openrouter timed out after {timeout_s}s", retryable=True)
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"openrouter returned {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc
            except httpx.HTTPError as exc:
                last = ProviderError(f"openrouter transport error: {exc}", retryable=True)
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(_retry_after(response, attempt))
        assert last is not None
        raise last

    def health(self) -> str:
        key = get_json(
            f"{self.base_url}/key",
            headers=self._headers(),
            timeout=10.0,
            unreachable="openrouter unreachable",
        )
        data = key.get("data") or {}
        limit = data.get("limit")
        remaining = data.get("limit_remaining")
        budget = (
            "no credit limit on this key"
            if limit is None
            else f"${remaining:.2f} of ${limit:.2f} left on this key"
        )
        return f"openrouter: pinned to {self.routing['provider']}; {budget}"
