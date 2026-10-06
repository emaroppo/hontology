"""Batched judging, aggregation rules, and the embedding cache off-switch.

The provider is mocked throughout — these assert the loop's behaviour, not a
model's, and a test suite must not need a GPU.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select

from hontology.db.models import Candidate, Document, Run, Verdict
from hontology.db.session import session_scope
from hontology.evalkit.config import ConfigError, normalize
from hontology.judge import prompts
from hontology.judge import run as judge_module
from hontology.judge.providers.base import Completion, ProviderError
from hontology.ontology import service

pytestmark = pytest.mark.requires_db


class FakeProvider:
    """Records every call so batching can be counted rather than assumed."""

    def __init__(self, responses: list[str] | Exception):
        self.responses = responses
        self.calls: list[str] = []

    def complete(self, *, system, prompt, config, want_json, want_reasoning, model):
        self.calls.append(prompt)
        if isinstance(self.responses, Exception):
            raise self.responses
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return Completion(
            text=self.responses[index],
            reasoning="",
            input_tokens=100,
            output_tokens=20,
            latency_s=1.0,
        )


@pytest.fixture
def batchable_run(tmp_path, monkeypatch):
    """One document with three selected concepts, and a readable body on disk."""
    from hontology.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "scrape_cache_dir", property(lambda self: tmp_path))
    (tmp_path / "batchhash01.txt").write_text("A crowd overturned cars downtown.")

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-batch", name="Batch")
        concepts = [
            service.create_concept(session, ontology.id, name=n, definition=f"{n}.").id
            for n in ("Riot", "Strike", "Flood")
        ]
        document = Document(
            url="https://batch.test/1", url_hash="batchhash01", body_path="batchhash01.txt"
        )
        session.add(document)
        session.flush()

        run = Run(
            name="batch",
            ontology_id=ontology.id,
            ontology_version="v1",
            config={},
            candidates_key="ck",
            judge_key="jk_ck",
            status="pending",
        )
        session.add(run)
        session.flush()

        for concept_id in concepts:
            session.add(
                Candidate(
                    run_id=run.id,
                    document_id=document.id,
                    concept_id=concept_id,
                    source="semantic",
                    score=0.7,
                    rank=1,
                    selected=True,
                )
            )
        ids = {
            "ontology": ontology.id,
            "concepts": concepts,
            "document": document.id,
            "run": run.id,
        }
    return ids


def batch_config(prompt_id: str = "strict_batch_v1") -> dict:
    return normalize({"judge": {"prompt_id": prompt_id}})


class TestBatchParsing:
    def test_every_requested_concept_gets_an_entry(self):
        """A model omitting one would otherwise shrink the denominator silently."""
        raw = json.dumps({"verdicts": [{"concept_id": 1, "matched": True, "confidence": 0.9}]})
        parsed = judge_module.parse_batch(raw, [1, 2, 3])

        assert set(parsed) == {1, 2, 3}
        assert parsed[1]["matched"] is True
        assert parsed[2]["matched"] is False
        assert parsed[2]["omitted"] is True
        assert parsed[1]["omitted"] is False

    def test_invented_concepts_are_ignored(self):
        raw = json.dumps(
            {
                "verdicts": [
                    {"concept_id": 99, "matched": True},
                    {"concept_id": 1, "matched": True},
                ]
            }
        )
        parsed = judge_module.parse_batch(raw, [1])
        assert set(parsed) == {1}

    def test_json_wrapped_in_prose_still_parses(self):
        raw = 'Sure:\n{"verdicts": [{"concept_id": 1, "matched": false}]}\nDone.'
        assert judge_module.parse_batch(raw, [1])[1]["matched"] is False

    def test_a_response_without_a_verdicts_array_raises(self):
        with pytest.raises(json.JSONDecodeError):
            judge_module.parse_batch('{"result": "ok"}', [1])

    def test_malformed_entries_are_skipped_not_fatal(self):
        raw = json.dumps({"verdicts": ["nonsense", {"concept_id": 1, "matched": True}]})
        parsed = judge_module.parse_batch(raw, [1])
        assert parsed[1]["matched"] is True


class TestBatchExecution:
    def test_one_call_serves_every_concept(self, batchable_run, monkeypatch):
        """The whole point: N concepts for one article cost one call, not N."""
        response = json.dumps(
            {
                "verdicts": [
                    {"concept_id": cid, "matched": i == 0, "confidence": 0.8, "country": "KE"}
                    for i, cid in enumerate(batchable_run["concepts"])
                ]
            }
        )
        provider = FakeProvider([response])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)

        with session_scope() as session:
            result = judge_module.judge_run(
                session,
                batchable_run["run"],
                config=batch_config(),
                judge_body_limit=2000,
            )

        assert len(provider.calls) == 1
        assert result["judged"] == 3
        assert result["matched"] == 1
        assert result["mode"] == "per-document"

    def test_the_prompt_lists_every_concept_once(self, batchable_run, monkeypatch):
        response = json.dumps({"verdicts": []})
        provider = FakeProvider([response])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)

        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )

        prompt = provider.calls[0]
        for concept_id in batchable_run["concepts"]:
            assert f"concept_id {concept_id}" in prompt
        # The body appears once, not once per concept — the cost saving.
        assert prompt.count("A crowd overturned cars downtown.") == 1

    def test_omissions_are_counted_so_an_overloaded_prompt_is_visible(
        self, batchable_run, monkeypatch
    ):
        response = json.dumps(
            {"verdicts": [{"concept_id": batchable_run["concepts"][0], "matched": True}]}
        )
        provider = FakeProvider([response])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)

        with session_scope() as session:
            result = judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )

        assert result["omitted_by_model"] == 2
        assert result["judged"] == 3  # still every pair, so the denominator holds

    def test_a_failed_call_errors_every_pair_in_the_document(self, batchable_run, monkeypatch):
        """Batching trades isolation: one bad response costs the whole group."""
        provider = FakeProvider(ProviderError("model exploded"))
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)

        with session_scope() as session:
            result = judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )

        assert result["errors"] == 3
        with session_scope() as session:
            verdicts = list(
                session.scalars(select(Verdict).where(Verdict.run_id == batchable_run["run"]))
            )
            assert len(verdicts) == 3
            assert all(v.error for v in verdicts)

    def test_latency_is_divided_across_the_group(self, batchable_run, monkeypatch):
        """Attributing the full call latency to each pair would triple the cost."""
        response = json.dumps(
            {
                "verdicts": [
                    {"concept_id": c, "matched": False} for c in batchable_run["concepts"]
                ]
            }
        )
        monkeypatch.setattr(judge_module, "get_provider", lambda name: FakeProvider([response]))

        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )
        with session_scope() as session:
            verdicts = list(
                session.scalars(select(Verdict).where(Verdict.run_id == batchable_run["run"]))
            )
            assert all(v.latency_s == pytest.approx(1.0 / 3) for v in verdicts)

    def test_resume_skips_pairs_already_judged(self, batchable_run, monkeypatch):
        response = json.dumps(
            {
                "verdicts": [
                    {"concept_id": c, "matched": False} for c in batchable_run["concepts"]
                ]
            }
        )
        monkeypatch.setattr(judge_module, "get_provider", lambda name: FakeProvider([response]))

        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )
        with session_scope() as session:
            again = judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )
        assert again["skipped_already_done"] == 3
        assert again["judged"] == 0

    def test_batch_records_samples_as_one(self, batchable_run, monkeypatch):
        """Repeating a batch call re-rolls every verdict together, so the votes
        are not independent and a vote fraction would overstate agreement."""
        response = json.dumps(
            {
                "verdicts": [
                    {"concept_id": c, "matched": False} for c in batchable_run["concepts"]
                ]
            }
        )
        monkeypatch.setattr(judge_module, "get_provider", lambda name: FakeProvider([response]))

        config = normalize({"judge": {"prompt_id": "strict_batch_v1", "samples": 5}})
        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=config, judge_body_limit=2000
            )
        with session_scope() as session:
            verdict = session.scalar(
                select(Verdict).where(Verdict.run_id == batchable_run["run"])
            )
            assert verdict.samples == 1


class TestPromptTemplateContract:
    def test_a_per_document_template_must_supply_build_batch(self):
        """The type advertised this capability before it existed; now it cannot."""
        with pytest.raises(ValueError, match="build_batch"):
            prompts.PromptTemplate(
                prompt_id="broken",
                mode=prompts.PER_DOCUMENT,
                system="x",
                build_pair=lambda d, c, b, limit: "",
            )

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown mode"):
            prompts.PromptTemplate(
                prompt_id="broken",
                mode="telepathy",
                system="x",
                build_pair=lambda d, c, b, limit: "",
            )

    def test_the_batch_template_is_registered_and_complete(self):
        template = prompts.get("strict_batch_v1")
        assert template.mode == prompts.PER_DOCUMENT
        assert template.build_batch is not None

    def test_batch_and_per_pair_share_guidance(self):
        """Only construction differs, so the two are comparable as an experiment."""
        assert prompts.get("strict_batch_v1").system == prompts.get("strict_v1").system


class TestAggregation:
    SAMPLES = [
        {"matched": True, "confidence": 0.9, "evidence": "a", "country": "KE"},
        {"matched": True, "confidence": 0.9, "evidence": "b", "country": "KE"},
        {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
    ]

    def test_majority_takes_the_plurality(self):
        result = judge_module.aggregate(self.SAMPLES, "majority")
        assert result["matched"] is True
        assert result["vote_fraction"] == pytest.approx(2 / 3)

    def test_unanimous_requires_every_sample(self):
        """The choice when a false positive is expensive."""
        result = judge_module.aggregate(self.SAMPLES, "unanimous")
        assert result["matched"] is False
        # The fraction reports agreement with the verdict actually reported.
        assert result["vote_fraction"] == pytest.approx(1 / 3)

    def test_any_matches_on_a_single_vote(self):
        samples = [
            {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
            {"matched": False, "confidence": 0.9, "evidence": "", "country": ""},
            {"matched": True, "confidence": 0.9, "evidence": "x", "country": "KE"},
        ]
        result = judge_module.aggregate(samples, "any")
        assert result["matched"] is True
        assert result["vote_fraction"] == pytest.approx(1 / 3)

    def test_a_single_sample_is_unaffected_by_the_rule(self):
        one = [{"matched": True, "confidence": 0.8, "evidence": "e", "country": "KE"}]
        for how in judge_module.AGGREGATIONS:
            assert judge_module.aggregate(one, how)["matched"] is True

    def test_an_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            judge_module.aggregate(self.SAMPLES, "vibes")

    def test_the_config_rejects_an_unknown_rule(self):
        """A config field that silently does nothing is a small lie."""
        with pytest.raises(ConfigError, match="aggregation"):
            normalize({"judge": {"aggregation": "vibes"}})

    def test_the_aggregation_choice_changes_the_run_key(self):
        """It changes results, so it is behaviour and belongs in the hash."""
        from hontology.evalkit.config import stage_keys

        majority = stage_keys(normalize({"judge": {"aggregation": "majority"}}), "v1")
        unanimous = stage_keys(normalize({"judge": {"aggregation": "unanimous"}}), "v1")
        assert majority["judge"] != unanimous["judge"]
        # But retrieval is untouched.
        assert majority["candidates"] == unanimous["candidates"]


class TestEmbeddingRefresh:
    def test_refresh_is_not_part_of_the_run_key(self):
        """Recomputing an identical vector gives an identical result, so it must
        not fork the artifact tree."""
        from hontology.evalkit.config import stage_keys

        baseline = stage_keys(normalize({}), "v1")
        # There is deliberately no config field for it; it is a runtime flag.
        assert "refresh" not in json.dumps(normalize({}))
        assert baseline == stage_keys(normalize({}), "v1")

    def test_ensure_embeddings_accepts_the_flag(self):
        import inspect

        from hontology.retrieve import embed

        assert "refresh" in inspect.signature(embed.ensure_embeddings).parameters
        assert "refresh" in inspect.signature(embed.embed_documents).parameters


class TestNarrowedJudging:
    """Judging one window at a time, and a budget spent on the strongest pairs."""

    def _score(self, run, scores: dict[int, float]) -> None:
        with session_scope() as session:
            for candidate in session.scalars(select(Candidate).where(Candidate.run_id == run)):
                candidate.score = scores[candidate.concept_id]

    def test_a_budget_takes_the_highest_scores_first(self, batchable_run, monkeypatch):
        first, second, third = batchable_run["concepts"]
        self._score(batchable_run["run"], {first: 0.5, second: 0.9, third: 0.7})
        provider = FakeProvider(['{"matched": false, "confidence": 0.5}'])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)

        with session_scope() as session:
            judge_module.judge_run(
                session,
                batchable_run["run"],
                config=normalize({}),
                judge_body_limit=2000,
                limit=2,
                by_score=True,
            )
        with session_scope() as session:
            judged = set(
                session.scalars(
                    select(Verdict.concept_id).where(Verdict.run_id == batchable_run["run"])
                )
            )
        assert judged == {second, third}

    def test_other_documents_are_left_alone(self, batchable_run, monkeypatch):
        provider = FakeProvider(['{"matched": false, "confidence": 0.5}'])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)
        with session_scope() as session:
            result = judge_module.judge_run(
                session,
                batchable_run["run"],
                config=normalize({}),
                judge_body_limit=2000,
                document_ids={batchable_run["document"] + 1},
            )
        assert result["total"] == 0
        assert provider.calls == []


class TestCostRecorded:
    def test_per_pair_verdicts_carry_their_tokens(self, batchable_run, monkeypatch):
        provider = FakeProvider(['{"matched": false, "confidence": 0.5}'])
        monkeypatch.setattr(judge_module, "get_provider", lambda name: provider)
        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=normalize({}), judge_body_limit=2000
            )
        with session_scope() as session:
            rows = session.execute(
                select(Verdict.input_tokens, Verdict.output_tokens).where(
                    Verdict.run_id == batchable_run["run"]
                )
            ).all()
        assert rows and all(row == (100, 20) for row in rows)

    def test_a_batched_call_splits_its_tokens_exactly(self, batchable_run, monkeypatch):
        """One call for three concepts is recorded as exactly one call's worth."""
        response = json.dumps(
            {
                "verdicts": [
                    {"concept_id": c, "matched": False} for c in batchable_run["concepts"]
                ]
            }
        )
        monkeypatch.setattr(judge_module, "get_provider", lambda name: FakeProvider([response]))
        with session_scope() as session:
            judge_module.judge_run(
                session, batchable_run["run"], config=batch_config(), judge_body_limit=2000
            )
        with session_scope() as session:
            total = session.scalar(
                select(func.sum(Verdict.input_tokens)).where(
                    Verdict.run_id == batchable_run["run"]
                )
            )
        # 100 split three ways: 34 + 33 + 33, so nothing is lost to rounding.
        assert total == 100
