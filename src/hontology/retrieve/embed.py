"""Embedding providers and content-addressed embedding storage.

Two details here are easy to get wrong and expensive to debug:

**Task-instruction prefixes.** Several embedding models are trained asymmetrically
and expect the *document* side and the *query* side to be prefixed differently.
Embedding both sides identically with such a model still produces plausible
vectors and plausible-looking cosine scores — it just retrieves noticeably worse,
with nothing to indicate why. The prefixes are therefore part of the provider,
not something a caller can forget.

**Content-addressed keys.** An embedding is stored under
``"<fields>@<hash of the exact text>"``. Keying on the field composition alone
means an edited definition keeps its old vector under the same key, and retrieval
silently keeps scoring wording that no longer exists. Hashing the text makes a
change miss the cache, and lets several versions coexist so an old run replays
against the vectors it actually used.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Embedding, EmbeddingModel
from hontology.judge.providers.base import ProviderError

# Per-model (document_prefix, query_prefix), as given on each model's card on
# Hugging Face (nomic-ai/nomic-embed-text-v1.5, mixedbread-ai/mxbai-embed-large-v1).
# Models not listed get no prefix, which is the correct default for symmetric
# models.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": (
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
}


def document_prefix(model: str) -> str:
    return _PREFIXES.get(model, ("", ""))[0]


def query_prefix(model: str) -> str:
    return _PREFIXES.get(model, ("", ""))[1]


def normalize_for_embedding(text: str) -> str:
    """Lowercase text that is entirely upper case.

    This is not cosmetic. Short ALL-CAPS strings make some embedding models
    collapse: feeding seven distinct CAMEO root labels ("PROTEST", "ASSAULT",
    "APPEAL", …) to nomic-embed-text returns only **three** distinct vectors, with
    a mean pairwise cosine of 0.97 — several are bit-for-bit identical. Retrieval
    then ranks essentially at random while looking perfectly healthy, because the
    scores are plausible numbers in the right range.

    Lowercasing the same seven labels yields seven distinct vectors with a mean
    pairwise cosine of 0.65. Mixed-case prose is unaffected, so this is safe to
    apply to every embedded string.
    """
    return text.lower() if text and not any(c.islower() for c in text if c.isalpha()) else text


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...

    def dimension(self, model: str) -> int: ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, host: str, *, batch_size: int = 64, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embed texts, batching so one request cannot grow unbounded."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
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
                raise ProviderError(
                    f"ollama embed transport error: {exc}", retryable=True
                ) from exc

            vectors = body.get("embeddings")
            if not vectors or len(vectors) != len(chunk):
                raise ProviderError(
                    f"ollama embed: sent {len(chunk)} texts to {model!r}, got "
                    f"{len(vectors or [])} vectors back "
                    f"({body.get('error') or 'no error given'})"
                )
            out.extend(vectors)
        return out

    def dimension(self, model: str) -> int:
        return len(self.embed(["dimension probe"], model=model)[0])


def get_provider(name: str, host: str) -> EmbeddingProvider:
    if name == "ollama":
        return OllamaEmbeddingProvider(host)
    raise ProviderError(f"unknown embedding provider {name!r}")


# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------

CONCEPT_FIELD_SETS = ("name", "definition", "name+definition", "name+definition+inclusion")


def concept_text(concept: Concept, fields: str) -> str:
    """Build the text embedded for a concept.

    Which fields are embedded is a real retrieval lever, not a detail: a concept
    whose name is vague but whose definition is precise retrieves very differently
    under ``name`` than under ``name+definition``.
    """
    name = (concept.name or "").strip()
    definition = (concept.definition or "").strip()
    inclusion = (concept.inclusion_criteria or "").strip()

    if fields == "name":
        return name
    if fields == "definition":
        return definition
    if fields == "name+definition":
        return f"{name}\n{definition}".strip()
    if fields == "name+definition+inclusion":
        return f"{name}\n{definition}\n{inclusion}".strip()
    raise ValueError(f"unknown concept field set {fields!r}")


def content_key(fields: str, text: str) -> str:
    """``"<fields>@<sha256[:12]>"`` — encodes both what was embedded and its text.

    Hashes the *normalized* text, so the key identifies exactly what the model
    saw rather than what the caller passed in.
    """
    digest = hashlib.sha256(normalize_for_embedding(text).encode("utf-8")).hexdigest()[:12]
    return f"{fields}@{digest}"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def get_or_create_model(
    session: Session, provider: EmbeddingProvider, model: str
) -> EmbeddingModel:
    key = f"{provider.name}/{model}"
    existing = session.scalar(select(EmbeddingModel).where(EmbeddingModel.key == key))
    if existing is not None:
        return existing

    row = EmbeddingModel(
        key=key,
        provider=provider.name,
        model_name=model,
        dim=provider.dimension(model),
        normalized=True,
    )
    session.add(row)
    session.flush()
    return row


def _existing_pairs(
    session: Session, model_id: int, object_type: str, keys: list[str]
) -> set[tuple[int, str]]:
    """Which ``(object_id, text_key)`` rows already exist.

    Keyed by the *pair*, not the key alone. Two objects can legitimately share
    identical text — two concepts with the same definition, or two codes with the
    same label — and checking the key alone would treat the second as already
    embedded, leaving it with no row of its own. Similarity would then attribute
    that vector to only one of them and silently drop the other.
    """
    if not keys:
        return set()
    rows = session.execute(
        select(Embedding.object_id, Embedding.text_key).where(
            Embedding.model_id == model_id,
            Embedding.object_type == object_type,
            Embedding.text_key.in_(keys),
        )
    )
    return {(int(r[0]), r[1]) for r in rows}


def ensure_embeddings(
    session: Session,
    provider: EmbeddingProvider,
    model: str,
    *,
    object_type: str,
    items: dict[int, str],
    fields: str,
    refresh: bool = False,
) -> tuple[int, dict[int, str]]:
    """Ensure a content-addressed embedding exists for each ``{id: text}`` item.

    Only missing vectors are computed, so re-running after editing one concept
    embeds one concept. Returns ``(model_id, {id: text_key})``.

    ``refresh`` bypasses the cache and recomputes everything. It exists for when
    the cache itself is the suspect — without it the only way to force a clean
    re-embed is deleting rows from the database by hand. It deliberately does
    *not* belong in the run config: recomputing an identical vector produces an
    identical result, so it must not fork the artifact tree.
    """
    model_row = get_or_create_model(session, provider, model)
    keys = {oid: content_key(fields, text) for oid, text in items.items()}

    present = (
        set()
        if refresh
        else _existing_pairs(session, model_row.id, object_type, list(set(keys.values())))
    )
    missing = sorted(oid for oid, key in keys.items() if (oid, key) not in present)

    if missing:
        prefix = document_prefix(model)
        texts = [prefix + normalize_for_embedding(items[oid]) for oid in missing]
        vectors = provider.embed(texts, model=model)
        if refresh:
            # Replace rather than duplicate: the unique constraint is on
            # (model, object_type, object_id, text_key).
            session.execute(
                delete(Embedding).where(
                    Embedding.model_id == model_row.id,
                    Embedding.object_type == object_type,
                    Embedding.object_id.in_(missing),
                )
            )
            session.flush()

        for oid, source_text, vector in zip(missing, texts, vectors, strict=True):
            session.add(
                Embedding(
                    model_id=model_row.id,
                    object_type=object_type,
                    object_id=oid,
                    text_key=keys[oid],
                    # The text as the model actually saw it, so a stored vector
                    # can always be traced back to its exact input.
                    source_text=source_text,
                    embedding=vector,
                )
            )
        session.flush()

    return model_row.id, keys


def embed_documents(
    session: Session,
    provider: EmbeddingProvider,
    model: str,
    items: dict[int, str],
    *,
    body_limit: int,
    refresh: bool = False,
) -> tuple[int, dict[int, str]]:
    """Ensure a cached embedding exists for each ``{document_id: body}``.

    Document bodies were previously embedded fresh on every run and thrown away.
    That is pure waste when iterating on *selection* parameters — min_score,
    max_k, pool_size change nothing about the text, yet each sweep cell re-embedded
    an identical corpus.

    The key includes ``body_limit`` because truncating at a different length
    produces genuinely different text, so a changed limit must miss the cache
    rather than silently reuse a vector built from more or less of the article.
    """
    return ensure_embeddings(
        session,
        provider,
        model,
        object_type="document",
        items=items,
        fields=f"body@{body_limit}",
        refresh=refresh,
    )


def embed_concepts(
    session: Session,
    provider: EmbeddingProvider,
    model: str,
    ontology_id: int,
    *,
    fields: str = "name+definition",
) -> tuple[int, dict[int, str]]:
    concepts = list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
    items = {c.id: concept_text(c, fields) for c in concepts}
    items = {oid: text for oid, text in items.items() if text}
    return ensure_embeddings(
        session, provider, model, object_type="concept", items=items, fields=fields
    )
