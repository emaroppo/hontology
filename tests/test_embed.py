"""Embedding text handling.

The normalization rule here was found by debugging, not by design: see
`normalize_for_embedding`. These tests exist so the collapse cannot come back
quietly, because its symptom is bad ranking with entirely healthy-looking scores.
"""

from __future__ import annotations

import pytest

from hontology.retrieve.embed import (
    content_key,
    normalize_for_embedding,
    query_prefix,
)


class TestNormalization:
    def test_all_caps_is_lowercased(self):
        """CAMEO root labels are all upper case, which makes models degenerate."""
        assert normalize_for_embedding("MAKE PUBLIC STATEMENT") == "make public statement"
        assert normalize_for_embedding("PROTEST") == "protest"

    def test_mixed_case_prose_is_untouched(self):
        text = "A commercial seaport halts vessel operations."
        assert normalize_for_embedding(text) == text

    def test_title_case_is_untouched(self):
        assert normalize_for_embedding("Protest violently, riot") == "Protest violently, riot"

    def test_caps_with_punctuation_and_digits_still_normalizes(self):
        assert normalize_for_embedding("USE UNCONVENTIONAL MASS VIOLENCE") == (
            "use unconventional mass violence"
        )
        assert normalize_for_embedding("REDUCE RELATIONS (14)") == "reduce relations (14)"

    def test_a_single_lowercase_letter_marks_it_as_mixed(self):
        assert normalize_for_embedding("PROTESTs") == "PROTESTs"

    def test_non_alphabetic_text_is_untouched(self):
        assert normalize_for_embedding("1451") == "1451"
        assert normalize_for_embedding("") == ""


class TestContentKey:
    def test_key_encodes_the_field_set(self):
        assert content_key("name+definition", "a riot").startswith("name+definition@")

    def test_different_text_gives_a_different_key(self):
        assert content_key("name", "a riot") != content_key("name", "a strike")

    def test_same_text_under_different_fields_gives_different_keys(self):
        assert content_key("name", "a riot") != content_key("definition", "a riot")

    def test_key_hashes_the_normalized_text(self):
        """Casing that the model never sees must not fork the cache."""
        assert content_key("code:root", "PROTEST") == content_key("code:root", "protest")


class TestPrefixes:
    def test_asymmetric_model_gets_a_query_prefix(self):
        assert query_prefix("nomic-embed-text") == "search_query: "

    def test_unknown_model_gets_no_prefix(self):
        assert query_prefix("some-other-model") == ""


@pytest.mark.requires_llm
def test_normalization_prevents_vector_collapse():
    """The actual failure: distinct labels collapsing onto identical vectors."""
    from hontology.config import get_settings
    from hontology.retrieve.embed import document_prefix, get_provider

    settings = get_settings()
    provider = get_provider("ollama")
    model = settings.default_embed_model
    labels = ["PROTEST", "ASSAULT", "APPEAL", "FIGHT", "COERCE", "YIELD", "DEMAND"]

    prefix = document_prefix(model)
    raw = provider.embed([prefix + label for label in labels], model=model)
    normalized = provider.embed(
        [prefix + normalize_for_embedding(label) for label in labels], model=model
    )

    def distinct(vectors: list[list[float]]) -> int:
        return len({tuple(v) for v in vectors})

    # The bug: fewer distinct vectors than inputs.
    assert distinct(raw) < len(labels)
    # The fix: one distinct vector per input.
    assert distinct(normalized) == len(labels)
