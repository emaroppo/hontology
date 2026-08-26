"""Judge parsing, aggregation and cutoff selection.

Pure logic, no model. These cover the ways a model's output goes wrong in
practice — wrapped JSON, out-of-range confidence, wrong types — because each one
is a silent corruption if it is not handled here.
"""

from __future__ import annotations

import json

import pytest

from hontology.judge.prompts import available, get
from hontology.judge.run import aggregate, parse_verdict
from hontology.retrieve.candidates import select_adaptive


class TestParseVerdict:
    def test_clean_json(self):
        result = parse_verdict(
            '{"matched": true, "confidence": 0.9, "country": "KE", "evidence": "quote"}'
        )
        assert result == {
            "matched": True,
            "confidence": 0.9,
            "country": "KE",
            "evidence": "quote",
        }

    def test_json_wrapped_in_prose(self):
        """Models add preamble even when told not to; discarding a good answer
        over a stray 'Here you go:' throws away real work."""
        raw = (
            'Sure! Here is the result:\n{"matched": false, "confidence": 0.2}\nHope that helps.'
        )
        assert parse_verdict(raw)["matched"] is False

    def test_json_in_a_code_fence(self):
        raw = '```json\n{"matched": true, "confidence": 0.7}\n```'
        assert parse_verdict(raw)["matched"] is True

    def test_confidence_is_clamped(self):
        assert parse_verdict('{"matched": true, "confidence": 1.7}')["confidence"] == 1.0
        assert parse_verdict('{"matched": true, "confidence": -3}')["confidence"] == 0.0

    def test_non_numeric_confidence_becomes_zero(self):
        assert parse_verdict('{"matched": true, "confidence": "high"}')["confidence"] == 0.0

    def test_missing_fields_default_safely(self):
        result = parse_verdict("{}")
        assert result["matched"] is False
        assert result["confidence"] == 0.0
        assert result["country"] == ""
        assert result["evidence"] == ""

    def test_country_is_normalized_to_upper(self):
        assert parse_verdict('{"matched": true, "country": "ke"}')["country"] == "KE"

    def test_null_fields_are_tolerated(self):
        result = parse_verdict('{"matched": true, "country": null, "evidence": null}')
        assert result["country"] == ""
        assert result["evidence"] == ""

    def test_unparseable_raises(self):
        """The caller records this as an error row rather than guessing."""
        with pytest.raises(json.JSONDecodeError):
            parse_verdict("I cannot answer that question.")


class TestAggregate:
    def test_single_sample_has_full_vote_fraction(self):
        result = aggregate(
            [{"matched": True, "confidence": 0.8, "evidence": "e", "country": "KE"}]
        )
        assert result["vote_fraction"] == 1.0
        assert result["confidence"] == 0.8

    def test_majority_wins_and_replaces_confidence(self):
        """A model answering yes 3 of 5 times is uncertain in a way its own
        stated 0.95 does not capture."""
        samples = [
            {"matched": True, "confidence": 0.95, "evidence": "a", "country": "KE"},
            {"matched": True, "confidence": 0.95, "evidence": "b", "country": "KE"},
            {"matched": True, "confidence": 0.95, "evidence": "c", "country": "KE"},
            {"matched": False, "confidence": 0.95, "evidence": "", "country": ""},
            {"matched": False, "confidence": 0.95, "evidence": "", "country": ""},
        ]
        result = aggregate(samples)
        assert result["matched"] is True
        assert result["vote_fraction"] == 0.6
        assert result["confidence"] == 0.6

    def test_minority_verdict_loses(self):
        samples = [
            {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
            {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
            {"matched": True, "confidence": 0.9, "evidence": "x", "country": "KE"},
        ]
        result = aggregate(samples)
        assert result["matched"] is False
        assert result["vote_fraction"] == pytest.approx(2 / 3)

    def test_evidence_comes_from_a_sample_that_voted_with_the_majority(self):
        samples = [
            {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
            {"matched": True, "confidence": 0.9, "evidence": "real quote", "country": "KE"},
            {"matched": True, "confidence": 0.9, "evidence": "other quote", "country": "KE"},
        ]
        result = aggregate(samples)
        assert result["matched"] is True
        assert result["evidence"] in {"real quote", "other quote"}


class TestAdaptiveSelection:
    def test_keeps_near_ties(self):
        ranked = [(1, 0.80), (2, 0.78), (3, 0.40)]
        chosen = select_adaptive(ranked, min_score=0.3, rel_margin=0.05, max_k=8)
        assert [cid for cid, _ in chosen] == [1, 2]

    def test_respects_the_floor(self):
        ranked = [(1, 0.20), (2, 0.19)]
        assert select_adaptive(ranked, min_score=0.45, rel_margin=0.05, max_k=8) == []

    def test_respects_max_k(self):
        ranked = [(i, 0.9) for i in range(10)]
        assert len(select_adaptive(ranked, min_score=0.3, rel_margin=0.05, max_k=3)) == 3

    def test_empty_pool(self):
        assert select_adaptive([], min_score=0.3, rel_margin=0.05, max_k=8) == []

    def test_cutoff_is_relative_to_this_document(self):
        """A weak-best document still gets its best match, unlike a flat threshold."""
        weak = [(1, 0.52), (2, 0.30)]
        chosen = select_adaptive(weak, min_score=0.45, rel_margin=0.05, max_k=8)
        assert [cid for cid, _ in chosen] == [1]


class TestPromptRegistry:
    def test_the_default_prompt_is_registered(self):
        assert "strict_v1" in available()

    def test_unknown_prompt_lists_what_exists(self):
        with pytest.raises(ValueError, match="registered:"):
            get("no_such_prompt")

    def test_strict_names_the_exclusion_cases(self):
        """These exclusions are the main precision lever; losing them is silent."""
        system = get("strict_v1").system
        for cue in ("NOT YET", "ONLY URGED", "FELL SHORT", "DENIED", "BACKGROUND"):
            assert cue in system

    def test_lenient_baseline_omits_them(self):
        """It exists to make the exclusions' contribution measurable."""
        assert "NOT YET" not in get("lenient_v1").system

    def test_variants_share_guidance_but_differ_in_order(self):
        assert get("concept_first_v1").system == get("strict_v1").system
        assert get("concept_first_v1").prompt_id != get("strict_v1").prompt_id
