"""Run configuration and stage-key composition.

The rules here decide what gets recomputed and what gets reused, and whether two
results are comparable at all. They are cheap to get subtly wrong and expensive
to notice, so each one is pinned.
"""

from __future__ import annotations

import pytest

from hontology.pipeline.runs.config import ConfigError, manifest, normalize, stage_keys

VERSION = "v3"


def keys(config: dict, version: str = VERSION) -> dict[str, str]:
    return stage_keys(normalize(config), version)


class TestDefaults:
    def test_empty_config_is_fully_defaulted(self):
        norm = normalize({})
        assert norm["candidates"]["source"] == "semantic"
        assert norm["judge"]["provider"] == "ollama"
        assert norm["judge"]["generation"]["temperature"] == 0.0

    def test_per_stage_body_limits_inherit_the_shared_one(self):
        norm = normalize({"common": {"body_limit": 4000}})
        assert norm["common"]["embed_body_limit"] == 4000
        assert norm["common"]["judge_body_limit"] == 4000

    def test_an_explicit_per_stage_limit_wins(self):
        norm = normalize({"common": {"body_limit": 4000, "judge_body_limit": 1000}})
        assert norm["common"]["embed_body_limit"] == 4000
        assert norm["common"]["judge_body_limit"] == 1000

    def test_nested_generation_settings_merge_rather_than_replace(self):
        norm = normalize({"judge": {"generation": {"seed": 7}}})
        assert norm["judge"]["generation"]["seed"] == 7
        # Untouched siblings survive.
        assert norm["judge"]["generation"]["temperature"] == 0.0

    @pytest.mark.parametrize(
        "config",
        [
            {"candidates": {"source": "telepathy"}},
            {"candidates": {"selection": "vibes"}},
            {"judge": {"samples": 0}},
        ],
    )
    def test_invalid_values_are_rejected(self, config):
        with pytest.raises(ConfigError):
            normalize(config)


class TestReuse:
    def test_identical_configs_share_both_keys(self):
        assert keys({}) == keys({})

    def test_changing_the_prompt_reuses_retrieval(self):
        """The property that makes prompt iteration affordable."""
        before = keys({"judge": {"prompt_id": "strict_v1"}})
        after = keys({"judge": {"prompt_id": "lenient_v1"}})

        assert before["candidates"] == after["candidates"]
        assert before["judge"] != after["judge"]

    def test_changing_the_judge_model_reuses_retrieval(self):
        before = keys({"judge": {"model": "model-a"}})
        after = keys({"judge": {"model": "model-b"}})
        assert before["candidates"] == after["candidates"]
        assert before["judge"] != after["judge"]

    def test_changing_retrieval_changes_both_keys(self):
        """The judge artifact is only meaningful for the candidates it judged."""
        before = keys({"candidates": {"embed_model": "model-a"}})
        after = keys({"candidates": {"embed_model": "model-b"}})

        assert before["candidates"] != after["candidates"]
        assert before["judge"] != after["judge"]

    def test_the_judge_key_embeds_the_candidates_key(self):
        composed = keys({})
        assert composed["judge"].endswith(f"_{composed['candidates']}")

    def test_an_unknown_retrieval_source_is_rejected(self):
        """`code` was removed as a retrieval source; the signal moved to the
        ingest filter, where it can decide what to fetch at all."""
        with pytest.raises(ConfigError):
            normalize({"candidates": {"source": "code"}})


class TestBehaviourVersusInfrastructure:
    @pytest.mark.parametrize(
        "infra",
        [
            {"ollama_host": "http://gpu-box:11434"},
            {"database_url": "postgresql://elsewhere/db"},
            {"api_key": "secret"},
            {"region": "eu-west-1"},
        ],
    )
    def test_infrastructure_never_changes_a_key(self, infra):
        """The same run served from another machine is the same run."""
        baseline = keys({})
        assert keys({"judge": dict(infra)}) == baseline
        assert keys({"candidates": dict(infra)}) == baseline

    def test_infrastructure_is_stripped_from_the_stored_config(self):
        norm = normalize({"judge": {"ollama_host": "http://gpu-box:11434"}})
        assert "ollama_host" not in norm["judge"]

    def test_provider_and_model_do_change_keys(self):
        """These are behaviour, not transport."""
        assert keys({"judge": {"provider": "anthropic"}}) != keys({})
        assert keys({"judge": {"model": "other"}}) != keys({})

    def test_name_and_description_do_not_change_keys(self):
        """Renaming an experiment must not orphan its artifacts."""
        assert keys({"name": "first"}) == keys({"name": "second", "description": "notes"})


class TestOntologyVersion:
    def test_a_different_ontology_version_changes_both_keys(self):
        before = keys({}, "v1")
        after = keys({}, "v2")
        assert before["candidates"] != after["candidates"]
        assert before["judge"] != after["judge"]

    def test_body_limits_are_scoped_to_their_own_stage(self):
        """An embed-only limit must not fork the judge's half of the key."""
        base = normalize({})
        embed_only = normalize({"common": {"embed_body_limit": 999}})

        base_keys = stage_keys(base, VERSION)
        changed = stage_keys(embed_only, VERSION)

        assert base_keys["candidates"] != changed["candidates"]
        # The judge half (before the underscore) is unchanged.
        assert changed["judge"].split("_")[0] == base_keys["judge"].split("_")[0]

    def test_a_judge_only_limit_leaves_retrieval_alone(self):
        base = keys({})
        changed = keys({"common": {"judge_body_limit": 999}})
        assert base["candidates"] == changed["candidates"]
        assert base["judge"] != changed["judge"]


class TestManifest:
    def test_manifest_records_infrastructure_without_hashing_it(self):
        norm = normalize({"name": "baseline"})
        record = manifest(norm, VERSION, infra={"ollama_host": "http://gpu-box:11434"})

        assert record["infra"]["ollama_host"] == "http://gpu-box:11434"
        assert record["keys"] == stage_keys(norm, VERSION)
        assert record["ontology_version"] == VERSION
