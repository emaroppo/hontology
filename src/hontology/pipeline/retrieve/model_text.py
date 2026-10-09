"""Text exactly as an embedding model should see it.

**Task-instruction prefixes.** Several embedding models are trained asymmetrically
and expect the *document* side and the *query* side to be prefixed differently.
Embedding both sides identically with such a model still produces plausible
vectors and plausible-looking cosine scores — it just retrieves noticeably worse,
with nothing to indicate why. The prefixes are therefore looked up from the
model, not something a caller can forget.

**Case.** Text entirely in upper case is lowercased first; see
`normalize_for_embedding` for why that is not cosmetic.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Per-model-family (document_prefix, query_prefix), as given on each model's card
# on Hugging Face (nomic-ai/nomic-embed-text-v1.5, mixedbread-ai/mxbai-embed-large-v1).
# Models not listed get no prefix, which is the correct default for symmetric
# models.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": (
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
}


def _prefixes(model: str) -> tuple[str, str]:
    """Look a model up by family, not exact name.

    The same weights go by many names — ``nomic-embed-text:latest`` in Ollama,
    ``nomic-embed-text-v1.5.Q8_0`` as a GGUF file served by llama.cpp — and an
    exact-match miss silently embeds without the prefix the model was trained on.
    """
    name = PurePosixPath(model).name.lower()
    for family, prefixes in _PREFIXES.items():
        if name.startswith(family):
            return prefixes
    return ("", "")


def document_prefix(model: str) -> str:
    return _prefixes(model)[0]


def query_prefix(model: str) -> str:
    return _prefixes(model)[1]


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
