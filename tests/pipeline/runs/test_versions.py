"""Stage versions: what changes each one, and what must not.

A version that moves when nothing relevant changed splits one stage into many;
one that stays put when something did merges different stages. Both make a
per-stage score meaningless, so each rule is pinned here.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from hontology.db.models import Code, Run
from hontology.db.session import session_scope
from hontology.ontology import portable, service
from hontology.pipeline.ingest.codes import cameo
from hontology.pipeline.judge import prompts
from hontology.pipeline.retrieve import similarity
from hontology.pipeline.runs import fingerprints, link_versions, runner, versions

pytestmark = pytest.mark.requires_db

CONFIG = {
    "candidates": {"selection": "adaptive", "min_score": 0.5, "max_k": 3},
    "judge": {"provider": "llamacpp", "model": "gemma", "prompt_id": "strict_v1"},
}


def _ontology(session, slug: str = "test-versions"):
    ontology = portable.import_ontology(
        session,
        {
            "export_version": 2,
            "slug": slug,
            "name": "Versions",
            "concepts": [
                {"name": "Disruption", "definition": "Something stops."},
                {"name": "Strike", "definition": "Workers stop."},
                {"name": "Flood", "definition": "Water rises."},
            ],
            "relations": [
                ["Strike", "subclass_of", "Disruption"],
                ["Flood", "subclass_of", "Disruption"],
            ],
        },
    )
    return ontology


def _run(session, ontology_id: int, config: dict = CONFIG) -> Run:
    return runner.create_run(session, ontology_id=ontology_id, config=config)


def _reword(session, ontology_id: int, name: str, definition: str) -> None:
    concept = next(c for c in service.list_concepts(session, ontology_id) if c.name == name)
    service.update_concept(session, concept.id, definition=definition)


def test_a_new_run_records_its_fingerprints():
    with session_scope() as session:
        run = _run(session, _ontology(session).id)
        recorded = run.manifest["versions"]
    assert recorded == {
        "retrieval_code": fingerprints.retrieval_code_hash(),
        "prompt": fingerprints.prompt_fingerprint("strict_v1"),
    }


def test_the_cutoff_is_not_part_of_the_retrieval_version():
    with session_scope() as session:
        ontology = _ontology(session)
        a = _run(session, ontology.id)
        looser = CONFIG | {"candidates": {"selection": "top-k", "top_k": 7}}
        b = _run(session, ontology.id, looser)
        assert (
            versions.retrieval_version(session, a)["version"]
            == versions.retrieval_version(session, b)["version"]
        )


def test_rewording_an_internal_class_moves_judging_but_not_retrieval():
    """Retrieval ranks leaves only; the judge may be asked about any class."""
    with session_scope() as session:
        ontology = _ontology(session)
        before = _run(session, ontology.id)
        _reword(session, ontology.id, "Disruption", "Anything stops.")
        after = _run(session, ontology.id)
        assert before.ontology_version != after.ontology_version
        assert (
            versions.retrieval_version(session, before)["version"]
            == versions.retrieval_version(session, after)["version"]
        )
        assert (
            versions.judge_version(session, before)["version"]
            != versions.judge_version(session, after)["version"]
        )


def test_rewording_a_leaf_moves_retrieval():
    with session_scope() as session:
        ontology = _ontology(session)
        before = _run(session, ontology.id)
        _reword(session, ontology.id, "Strike", "Dockworkers stop.")
        after = _run(session, ontology.id)
        assert (
            versions.retrieval_version(session, before)["version"]
            != versions.retrieval_version(session, after)["version"]
        )


def test_a_run_that_reused_candidates_has_the_sources_retrieval():
    with session_scope() as session:
        ontology = _ontology(session)
        source = _run(session, ontology.id)
        arm = _run(session, ontology.id, CONFIG | {"common": {"embed_body_limit": 999}})
        arm.manifest = arm.manifest | {"sample": {"candidates_from": source.id}}
        session.flush()
        version = versions.retrieval_version(session, arm)
        assert version["source_run"] == source.id
        assert version["version"] == versions.retrieval_version(session, source)["version"]


def test_editing_a_prompts_text_changes_its_fingerprint(monkeypatch):
    # Fingerprints are cached per process, since templates are code; an edit in
    # place within one process has to clear the cache to be seen.
    fingerprints.prompt_fingerprint.cache_clear()
    template = prompts.get("strict_v1")
    before = fingerprints.prompt_fingerprint("strict_v1")
    monkeypatch.setitem(
        prompts._REGISTRY,
        "strict_v1",
        prompts.PromptTemplate(
            prompt_id="strict_v1",
            mode=template.mode,
            system=template.system + " Be careful.",
            build_pair=template.build_pair,
        ),
    )
    fingerprints.prompt_fingerprint.cache_clear()
    try:
        assert fingerprints.prompt_fingerprint("strict_v1") != before
    finally:
        fingerprints.prompt_fingerprint.cache_clear()


def test_link_snapshots_mint_only_on_change():
    with session_scope() as session:
        ontology = _ontology(session)
        assert link_versions.resolve_links(session, ontology.id) is None
        result = cameo.load_codes(session, cameo.parse_lookup("14\tPROTEST\n145\tRiot\n"))
        codes = {
            c.code: c.id
            for c in session.scalars(select(Code).where(Code.system_id == result["system_id"]))
        }
        strike = next(
            c for c in service.list_concepts(session, ontology.id) if c.name == "Strike"
        )
        similarity.set_link(session, strike.id, codes["14"], linked=True)
        first = link_versions.resolve_links(session, ontology.id)
        again = link_versions.resolve_links(session, ontology.id)
        similarity.set_link(session, strike.id, codes["145"], linked=True)
        second = link_versions.resolve_links(session, ontology.id)
        assert (first.version, again.version, second.version) == ("f1", "f1", "f2")
        assert link_versions.link_snapshot(session, ontology.id, "f1") == [
            [strike.id, "cameo", "14"]
        ]


def test_a_run_notes_each_filter_version_once():
    with session_scope() as session:
        run = _run(session, _ontology(session).id)
        for version in ("f1", "f1", "f2"):
            versions.note_filter(run, version)
        assert versions.filter_versions(run) == ["f1", "f2"]
        assert run.manifest["versions"]["prompt"]  # fingerprints are kept
