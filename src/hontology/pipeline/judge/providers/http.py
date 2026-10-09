"""HTTP plumbing the providers share: transport failures become ProviderErrors,
and OpenAI-style chat requests are built and read the same way everywhere."""

from __future__ import annotations

import time

import httpx

from hontology.pipeline.judge.providers.base import Completion, GenerationConfig, ProviderError


def get_json(
    url: str, *, unreachable: str, headers: dict | None = None, timeout: float = 5.0
) -> dict:
    """GET a JSON body; any failure reads ``<unreachable>: <error>``."""
    try:
        response = httpx.get(url, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderError(f"{unreachable}: {exc}", retryable=True) from exc
    return response.json()


def post_json(
    url: str, payload: dict, *, label: str, timeout: float, headers: dict | None = None
) -> dict:
    """POST a request and return its JSON reply, naming the backend in errors."""
    try:
        response = httpx.post(url, json=payload, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{label} timed out after {timeout}s", retryable=True) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise ProviderError(
            f"{label} returned {status}: {exc.response.text[:300]}",
            retryable=status >= 500,
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"{label} transport error: {exc}", retryable=True) from exc


def chat_payload(
    model: str, system: str | None, prompt: str, config: GenerationConfig, want_json: bool
) -> dict:
    """The parts of an OpenAI-style chat request every backend sends alike."""
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    payload: dict = {"model": model, "messages": messages, "temperature": config.temperature}
    if config.max_output_tokens is not None:
        payload["max_tokens"] = config.max_output_tokens
    if config.seed is not None:
        payload["seed"] = config.seed
    if want_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def read_chat(
    body: dict,
    *,
    label: str,
    model: str,
    config: GenerationConfig,
    want_json: bool,
    want_reasoning: bool,
    reasoning_key: str,
    started: float,
) -> Completion:
    """A Completion from an OpenAI-style reply, refusing an over-budget or empty one."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    stop_reason = choice.get("finish_reason")
    usage = body.get("usage") or {}

    total = usage.get("total_tokens")
    if total is not None and total > config.context_window:
        raise ProviderError(
            f"{label} call used {total} tokens, over the configured "
            f"context_window of {config.context_window}"
        )
    if not text.strip():
        # finish_reason "length" with reasoning but no answer means the
        # token budget ran out mid-thought.
        raise ProviderError(
            f"{label} model {model!r} returned an empty response "
            f"(finish_reason={stop_reason}, think={want_reasoning}, json={want_json})"
        )
    return Completion(
        text=text,
        reasoning=message.get(reasoning_key) or "",
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        latency_s=time.monotonic() - started,
        stop_reason=stop_reason,
    )
