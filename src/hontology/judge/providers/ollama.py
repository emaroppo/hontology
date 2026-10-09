"""Ollama-backed completion provider.

Two behaviors of the Ollama HTTP API drive the shape of this module, and both
cost real debugging time to discover, so they are handled explicitly here rather
than left for a caller to trip over:

**1. Structured output and thinking are mutually exclusive.** Sending
``format="json"`` together with ``think=true`` to a reasoning model returns a
*successful* response whose ``response`` field is empty — the grammar and the
thinking channel fight, and the answer is lost. The failure surfaces downstream
as a JSON parse error on empty input, which points at the parser instead of the
request. So when reasoning is requested the grammar is dropped and the model is
asked for JSON in the prompt instead, with the parser picking up the slack.

**2. Token counts live on the response envelope**, as ``prompt_eval_count`` and
``eval_count``. They are the only cost signal available for a local model, so
they are read here and carried on every :class:`Completion`.
"""

from __future__ import annotations

import time

from hontology.judge.providers.base import (
    ChatProvider,
    Completion,
    GenerationConfig,
    ProviderError,
)
from hontology.judge.providers.http import get_json, post_json


class OllamaChatProvider(ChatProvider):
    name = "ollama"

    def __init__(self, host: str) -> None:
        self.host = host.rstrip("/")

    def _options(self, config: GenerationConfig) -> dict:
        options: dict = {
            "temperature": config.temperature,
            "num_ctx": config.context_window,
        }
        if config.max_output_tokens is not None:
            options["num_predict"] = config.max_output_tokens
        if config.seed is not None:
            options["seed"] = config.seed
        return options

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
            raise ProviderError("ollama provider requires an explicit model")

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": want_reasoning,
            "options": self._options(config),
        }
        if system:
            # Goes into the chat template's system slot, which models weight more
            # heavily than the same text inlined at the top of the user prompt.
            payload["system"] = system
        if want_json and not want_reasoning:
            # See the module docstring: this combination is only safe when
            # thinking is off.
            payload["format"] = "json"

        started = time.monotonic()
        body = post_json(
            f"{self.host}/api/generate", payload, label="ollama", timeout=config.timeout_s
        )

        text = body.get("response") or ""
        reasoning = body.get("thinking") or ""

        if not text.strip():
            # Most often the grammar/thinking clash above, but that is guarded
            # now, so an empty body here means the model produced nothing usable.
            raise ProviderError(
                f"ollama model {model!r} returned an empty response "
                f"(think={want_reasoning}, json={want_json and not want_reasoning})"
            )

        return Completion(
            text=text,
            reasoning=reasoning,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            latency_s=time.monotonic() - started,
            stop_reason=body.get("done_reason"),
        )

    def health(self) -> str:
        tags = get_json(
            f"{self.host}/api/tags", unreachable=f"ollama unreachable at {self.host}"
        )
        names = [m.get("name", "?") for m in tags.get("models", [])]
        return f"ollama at {self.host}: {len(names)} model(s) available"
