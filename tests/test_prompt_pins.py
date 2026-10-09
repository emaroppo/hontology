"""A prompt id stands for one wording, for good.

Verdicts are recorded under a prompt id, so rewording an id in place would
change what every one of them means without a trace. Each released id is
pinned to the fingerprint of its rendered text; judging checks the pin, and a
run is never continued under wording other than the one it was created with.
"""

from __future__ import annotations

import pytest

from hontology.db.models import Run
from hontology.evalkit import versions
from hontology.judge import prompts
from hontology.judge.run import PromptChanged, check_wording


def test_every_released_prompt_is_pinned_to_its_wording():
    current = {pid: versions.prompt_fingerprint(pid) for pid in prompts.available()}
    assert current == prompts.PINS, (
        "prompts.lock.json does not match the registered prompts. A new prompt id "
        "needs its fingerprint pinned; a pinned id whose fingerprint changed was "
        "edited in place, which is not allowed: register the new wording under a "
        f"new id instead. Current fingerprints: {current}"
    )


def _run(prompt: str | None) -> Run:
    return Run(id=7, manifest={"versions": {"prompt": prompt}} if prompt else None)


def test_a_pinned_prompt_whose_wording_drifted_is_refused(monkeypatch):
    monkeypatch.setitem(prompts.PINS, "strict_v1", "000000000000")
    with pytest.raises(PromptChanged, match="edited in place"):
        check_wording(_run(None), "strict_v1")


def test_a_run_is_not_continued_under_other_wording():
    with pytest.raises(PromptChanged, match="mix two wordings"):
        check_wording(_run("000000000000"), "strict_v1")


def test_matching_wording_passes():
    check_wording(_run(versions.prompt_fingerprint("strict_v1")), "strict_v1")
    check_wording(_run(None), "strict_v1")  # a run from before fingerprints


def test_an_unpinned_prompt_can_be_tried(monkeypatch):
    """A template being developed has no pin yet; only shipping one needs it."""
    monkeypatch.setitem(prompts._REGISTRY, "draft_v1", prompts.get("strict_v1"))
    check_wording(_run(None), "draft_v1")
