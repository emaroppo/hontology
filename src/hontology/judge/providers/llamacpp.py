"""llama.cpp-backed completion provider, over its OpenAI-compatible API.

``llama-server`` speaks ``/v1/chat/completions``, but three of its behaviors
differ from what an OpenAI client would assume, and each one would let a run's
recorded config describe something other than what actually ran:

**1. The model name is not checked.** A server loads one model and answers any
``model`` string with it. A run configured for model A against a server holding
model B would succeed and be recorded as A. So the requested name is resolved
against ``/v1/models`` before the first call, and a mismatch is refused. The
server's id is usually a file path, which is a host detail rather than behavior,
so the file's stem (``gemma-4-26B-A4B-it-Q8_0`` for ``/models/gemma-4-26B-A4B-it-Q8_0.gguf``)
is accepted as the portable name.

**2. Thinking is on unless switched off.** Reasoning models served through their
chat template think by default, so omitting the switch turns ``think: false``
into a lie. ``enable_thinking`` is therefore always sent, in both directions.
Unlike Ollama, a JSON grammar and a thinking channel coexist here — the grammar
constrains only the answer — so JSON mode stays on when reasoning is requested.

**3. The context window is fixed at server start.** ``num_ctx`` has no
per-request equivalent. ``context_window`` is enforced as a ceiling instead: a
server whose per-slot context is smaller is refused up front, and a call whose
tokens exceed the configured window is an error rather than a silent success on
more context than the run claims to have used.
"""

from __future__ import annotations

import time
from pathlib import PurePosixPath

import httpx

from hontology.judge.providers.base import (
    ChatProvider,
    Completion,
    GenerationConfig,
    ProviderError,
)


def headers(api_key: str | None) -> dict[str, str]:
    """Bearer auth for a server started with ``--api-key``; nothing otherwise."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def model_stem(model_id: str) -> str:
    """The portable part of a served model id: its file name, minus ``.gguf``."""
    name = PurePosixPath(model_id).name
    return name[: -len(".gguf")] if name.endswith(".gguf") else name


def served_models(host: str, api_key: str | None = None) -> list[dict]:
    try:
        response = httpx.get(f"{host}/v1/models", headers=headers(api_key), timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderError(f"llama.cpp unreachable at {host}: {exc}", retryable=True) from exc
    return response.json().get("data", [])


def resolve_model(host: str, model: str, api_key: str | None = None) -> str:
    """Return the server's id for ``model``, or refuse if it is not being served.

    Matches the id, any alias, or the stem of either.
    """
    if not model:
        raise ProviderError("llama.cpp provider requires an explicit model")
    served = served_models(host, api_key)
    for entry in served:
        names = [entry.get("id", ""), *(entry.get("aliases") or [])]
        if model in names or model in {model_stem(n) for n in names}:
            return entry["id"]
    available = ", ".join(sorted(model_stem(e.get("id", "?")) for e in served)) or "none"
    raise ProviderError(f"llama.cpp at {host} is not serving {model!r} (serving: {available})")


class LlamaCppChatProvider(ChatProvider):
    name = "llamacpp"

    def __init__(self, host: str, api_key: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        # Requested name -> server id. Resolved once per provider, since the
        # server cannot swap models without a restart.
        self._resolved: dict[str, str] = {}
        self._slot_context: int | None = None

    def _model_id(self, model: str) -> str:
        if model not in self._resolved:
            self._resolved[model] = resolve_model(self.host, model, self.api_key)
        return self._resolved[model]

    def _check_context(self, config: GenerationConfig) -> None:
        if self._slot_context is None:
            try:
                response = httpx.get(
                    f"{self.host}/props", headers=headers(self.api_key), timeout=5.0
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"llama.cpp props unavailable at {self.host}: {exc}", retryable=True
                ) from exc
            self._slot_context = response.json()["default_generation_settings"]["n_ctx"]
        if self._slot_context < config.context_window:
            raise ProviderError(
                f"llama.cpp at {self.host} has {self._slot_context} tokens of context "
                f"per slot, below the configured context_window of "
                f"{config.context_window}; restart it with a larger -c or lower "
                f"context_window"
            )

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
        model_id = self._model_id(model)
        self._check_context(config)

        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "temperature": config.temperature,
            # See the module docstring: sent either way, never left to the default.
            "chat_template_kwargs": {"enable_thinking": want_reasoning},
        }
        if config.max_output_tokens is not None:
            payload["max_tokens"] = config.max_output_tokens
        if config.seed is not None:
            payload["seed"] = config.seed
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            response = httpx.post(
                f"{self.host}/v1/chat/completions",
                json=payload,
                headers=headers(self.api_key),
                timeout=config.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"llama.cpp timed out after {config.timeout_s}s", retryable=True
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise ProviderError(
                f"llama.cpp returned {status}: {exc.response.text[:300]}",
                retryable=status >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"llama.cpp transport error: {exc}", retryable=True) from exc

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        stop_reason = choice.get("finish_reason")
        usage = body.get("usage") or {}

        total = usage.get("total_tokens")
        if total is not None and total > config.context_window:
            raise ProviderError(
                f"llama.cpp call used {total} tokens, over the configured "
                f"context_window of {config.context_window}"
            )
        if not text.strip():
            # finish_reason "length" with reasoning but no answer means the
            # token budget ran out mid-thought.
            raise ProviderError(
                f"llama.cpp model {model!r} returned an empty response "
                f"(finish_reason={stop_reason}, think={want_reasoning}, json={want_json})"
            )

        return Completion(
            text=text,
            reasoning=reasoning,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_s=time.monotonic() - started,
            stop_reason=stop_reason,
        )

    def health(self) -> str:
        served = served_models(self.host, self.api_key)
        names = ", ".join(model_stem(m.get("id", "?")) for m in served)
        return f"llama.cpp at {self.host}: {len(served)} model(s) available ({names})"

    def available_models(self) -> list[str]:
        return sorted(
            model_stem(m.get("id", "")) for m in served_models(self.host, self.api_key)
        )
