"""Embedding providers: the services that turn text into vectors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from hontology.pipeline.judge.providers import llamacpp
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
