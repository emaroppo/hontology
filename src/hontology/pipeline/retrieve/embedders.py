"""Embedding providers: the services that turn text into vectors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from hontology.pipeline.judge.providers import llamacpp, openrouter
from hontology.pipeline.judge.providers.base import ProviderError


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...

    def dimension(self, model: str) -> int: ...


class _BatchedEmbedder(EmbeddingProvider):
    batch_size: int

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embed texts, batching so one request cannot grow unbounded."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_chunk(texts[start : start + self.batch_size], model))
        return out

    def _embed_chunk(self, chunk: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError

    def dimension(self, model: str) -> int:
        return len(self.embed(["dimension probe"], model=model)[0])


class OllamaEmbeddingProvider(_BatchedEmbedder):
    name = "ollama"

    def __init__(self, host: str, *, batch_size: int = 64, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def _embed_chunk(self, chunk: list[str], model: str) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.host}/api/embed",
                # truncate: clip inputs that exceed the context window rather
                # than returning 400, so one token-dense article cannot kill
                # a whole run. Character limits upstream are the real control;
                # this is the backstop.
                json={"model": model, "input": chunk, "truncate": True},
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"ollama embed {exc.response.status_code} for {model!r}: "
                f"{exc.response.text[:300]}",
                retryable=exc.response.status_code >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama embed transport error: {exc}", retryable=True) from exc

        vectors = body.get("embeddings")
        if not vectors or len(vectors) != len(chunk):
            raise ProviderError(
                f"ollama embed: sent {len(chunk)} texts to {model!r}, got "
                f"{len(vectors or [])} vectors back "
                f"({body.get('error') or 'no error given'})"
            )
        return vectors


class LlamaCppEmbeddingProvider(_BatchedEmbedder):
    """``/v1/embeddings`` on a ``llama-server`` started with ``--embeddings``.

    As with the chat provider, the requested model is checked against what the
    server actually holds: an embedding server answers any model name with its
    one model, and vectors filed under the wrong model name would poison the
    cache for every later run that trusts the key.
    """

    name = "llamacpp"

    def __init__(
        self,
        host: str,
        api_key: str | None = None,
        *,
        batch_size: int = 32,
        timeout: float = 300.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self._resolved: dict[str, str] = {}

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if model not in self._resolved:
            self._resolved[model] = llamacpp.resolve_model(self.host, model, self.api_key)
        return super().embed(texts, model=model)

    def _embed_chunk(self, chunk: list[str], model: str) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.host}/v1/embeddings",
                json={"model": self._resolved[model], "input": chunk},
                headers=llamacpp.headers(self.api_key),
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            # 501 is a server started without --embeddings, and the body
            # says so; a too-long input is a 500 naming the batch size.
            raise ProviderError(
                f"llama.cpp embed {exc.response.status_code} for {model!r}: "
                f"{exc.response.text[:300]}",
                retryable=exc.response.status_code >= 500 and exc.response.status_code != 501,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"llama.cpp embed transport error: {exc}", retryable=True
            ) from exc

        data = body.get("data") or []
        if len(data) != len(chunk):
            raise ProviderError(
                f"llama.cpp embed: sent {len(chunk)} texts to {model!r}, got "
                f"{len(data)} vectors back"
            )
        # The API carries an index per vector; order by it rather than trust
        # the response order.
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


class OpenRouterEmbeddingProvider(_BatchedEmbedder):
    """``/embeddings`` on OpenRouter, from one pinned host.

    Hosts of one embedding model can run it at different precisions, and their
    vectors differ, so the host is pinned as for the hosted judge and a reply
    from any other host is refused. The pinned host is also part of the
    provider's name, so vectors from two hosts never share a cache entry.
    """

    def __init__(
        self,
        api_key: str | None,
        routing: dict | None,
        base_url: str = openrouter.DEFAULT_BASE_URL,
        *,
        batch_size: int = 64,
        timeout: float = 120.0,
        sleep=None,
    ) -> None:
        if not api_key:
            raise ProviderError("openrouter needs HONTOLOGY_OPENROUTER_API_KEY")
        if not routing or not routing.get("provider"):
            raise ProviderError(
                "openrouter embeddings need candidates.embed_routing.provider: pin the host"
            )
        self.api_key, self.routing = api_key, routing
        self.base_url = base_url.rstrip("/")
        self.name = f"openrouter@{str(routing['provider']).lower()}"
        self.batch_size, self.timeout = batch_size, timeout
        self._sleep = sleep

    def _embed_chunk(self, chunk: list[str], model: str) -> list[list[float]]:
        body = openrouter.post(
            f"{self.base_url}/embeddings",
            {
                "model": model,
                "input": chunk,
                "provider": openrouter.routing_payload(self.routing),
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_s=self.timeout,
            **({"sleep": self._sleep} if self._sleep else {}),
        )
        openrouter.check_served_by(body, self.routing, model)
        data = body.get("data") or []
        if len(data) != len(chunk):
            raise ProviderError(
                f"openrouter embed: sent {len(chunk)} texts to {model!r}, got "
                f"{len(data)} vectors back"
            )
        return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
