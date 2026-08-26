"""Curated concept↔code links, and what a recompute is allowed to touch.

The rule under test: automatically proposed links carry the run that proposed
them and are rebuilt on every recompute; hand-made links have a NULL run id and
must survive untouched. Losing that distinction quietly discards every manual
correction, which is the kind of damage nobody notices until the work is gone.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from hontology.db.models import Code, ConceptCode, EmbeddingModel, SimilarityRun
from hontology.db.session import session_scope
from hontology.ingest import cameo
from hontology.ontology import service
from hontology.retrieve import similarity

pytestmark = pytest.mark.requires_db

SAMPLE = "14\tPROTEST\n145\tProtest violently, riot\n1451\tEngage in political dissent, riot\n"


@pytest.fixture
def fixture_ids():
    """An ontology with two concepts, and the CAMEO sample loaded."""

    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-sim", name="Sim")
        riot = service.create_concept(session, ontology.id, name="Riot", definition="A riot.")
        strike = service.create_concept(
            session, ontology.id, name="Strike", definition="A strike."
        )
        result = cameo.load_codes(session, cameo.parse_lookup(SAMPLE))
        codes = {
            c.code: c.id
            for c in session.scalars(select(Code).where(Code.system_id == result["system_id"]))
        }
        ids = {
            "ontology": ontology.id,
            "system": result["system_id"],
            "riot": riot.id,
            "strike": strike.id,
            "codes": codes,
        }
    yield ids


def make_run(session, level: str = "root") -> int:
    """A real similarity_runs row.

    The association table has a real foreign key to it, so a stand-in integer is
    correctly rejected — the schema will not let a link claim a run that never
    happened.
    """
    model = session.scalar(select(EmbeddingModel).where(EmbeddingModel.key == "test/fake"))
    if model is None:
        model = EmbeddingModel(key="test/fake", provider="test", model_name="fake", dim=3)
        session.add(model)
        session.flush()

    run = SimilarityRun(
        model_id=model.id,
        source_object_type="concept",
        source_text_key="name+definition",
        target_object_type=f"code:{level}",
        target_text_key=f"code:{level}",
    )
    session.add(run)
    session.flush()
    return run.id


def links_for(session, concept_id: int) -> dict[int, int | None]:
    """``{code_id: similarity_run_id}`` for a concept."""
    return {
        row.code_id: row.similarity_run_id
        for row in session.scalars(
            select(ConceptCode).where(ConceptCode.concept_id == concept_id)
        )
    }


def test_manual_link_has_no_run_id(fixture_ids):
    code_id = fixture_ids["codes"]["145"]
    with session_scope() as session:
        similarity.set_link(session, fixture_ids["riot"], code_id, linked=True)
    with session_scope() as session:
        assert links_for(session, fixture_ids["riot"]) == {code_id: None}


def test_recompute_replaces_auto_links_but_keeps_manual_ones(fixture_ids):
    codes = fixture_ids["codes"]
    riot, strike = fixture_ids["riot"], fixture_ids["strike"]

    # A human links Riot→145 by hand.
    with session_scope() as session:
        similarity.set_link(session, riot, codes["145"], linked=True)

    # A first run proposes Strike→14.
    with session_scope() as session:
        run_a = make_run(session)
        similarity._apply_links(
            session,
            run_a,
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {(strike, codes["14"]): 0.7},
        )

    with session_scope() as session:
        assert links_for(session, strike) == {codes["14"]: run_a}
        assert links_for(session, riot) == {codes["145"]: None}

    # A second run proposes something different at the same level.
    with session_scope() as session:
        run_b = make_run(session)
        similarity._apply_links(
            session,
            run_b,
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {(riot, codes["14"]): 0.8},
        )

    with session_scope() as session:
        # The stale auto link is gone, the new one is in.
        assert links_for(session, strike) == {}
        riot_links = links_for(session, riot)
        assert riot_links[codes["14"]] == run_b
        # The hand-made link is untouched by both runs.
        assert riot_links[codes["145"]] is None


def test_recompute_does_not_duplicate_an_existing_manual_pair(fixture_ids):
    codes = fixture_ids["codes"]
    riot = fixture_ids["riot"]

    with session_scope() as session:
        similarity.set_link(session, riot, codes["14"], linked=True)

    # The run proposes the very pair a human already asserted.
    with session_scope() as session:
        added, manual = similarity._apply_links(
            session,
            make_run(session),
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {(riot, codes["14"]): 0.9},
        )
        assert added == 0
        assert manual == 1

    with session_scope() as session:
        # Still exactly one row, still manual.
        assert links_for(session, riot) == {codes["14"]: None}


def test_confirming_a_proposal_promotes_it_to_manual(fixture_ids):
    """Ticking a suggestion in the UI must protect it from the next recompute."""
    codes = fixture_ids["codes"]
    riot = fixture_ids["riot"]

    with session_scope() as session:
        run_id = make_run(session)
        similarity._apply_links(
            session,
            run_id,
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {(riot, codes["14"]): 0.6},
        )
    with session_scope() as session:
        assert links_for(session, riot) == {codes["14"]: run_id}
        similarity.set_link(session, riot, codes["14"], linked=True)

    with session_scope() as session:
        assert links_for(session, riot) == {codes["14"]: None}
        # A later run that no longer proposes it cannot remove it now.
        similarity._apply_links(
            session,
            make_run(session),
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {},
        )
    with session_scope() as session:
        assert links_for(session, riot) == {codes["14"]: None}


def test_unticking_removes_a_link(fixture_ids):
    codes = fixture_ids["codes"]
    riot = fixture_ids["riot"]
    with session_scope() as session:
        similarity.set_link(session, riot, codes["14"], linked=True)
    with session_scope() as session:
        similarity.set_link(session, riot, codes["14"], linked=False)
    with session_scope() as session:
        assert links_for(session, riot) == {}


def test_recompute_is_scoped_to_the_level_it_ran_on(fixture_ids):
    """A root-level run must not disturb event-level proposals."""
    codes = fixture_ids["codes"]
    riot = fixture_ids["riot"]

    with session_scope() as session:
        run_event = make_run(session, "event")
        similarity._apply_links(
            session,
            run_event,
            fixture_ids["ontology"],
            fixture_ids["system"],
            "event",
            {(riot, codes["1451"]): 0.75},
        )
    with session_scope() as session:
        run_root = make_run(session, "root")
        similarity._apply_links(
            session,
            run_root,
            fixture_ids["ontology"],
            fixture_ids["system"],
            "root",
            {(riot, codes["14"]): 0.65},
        )

    with session_scope() as session:
        assert links_for(session, riot) == {codes["1451"]: run_event, codes["14"]: run_root}


def test_adaptive_selection_keeps_near_ties_per_concept():
    """A flat threshold suits concepts unevenly; adaptive is relative to each."""

    class Row:
        def __init__(self, concept_id, code_id, score):
            self.concept_id, self.code_id, self.score = concept_id, code_id, score

    pairs = [
        # Concept 1 has a strong best and one near-tie.
        Row(1, 10, 0.80),
        Row(1, 11, 0.78),
        Row(1, 12, 0.40),
        # Concept 2's best is weaker but still its best.
        Row(2, 10, 0.52),
        Row(2, 11, 0.30),
    ]
    selected = similarity._select_adaptive(pairs, min_score=0.25, rel_margin=0.05, max_k=15)

    assert (1, 10) in selected and (1, 11) in selected
    assert (1, 12) not in selected  # outside the margin
    assert (2, 10) in selected  # kept despite scoring below concept 1's cutoff
    assert (2, 11) not in selected


def test_adaptive_selection_drops_a_concept_with_no_decent_match():
    class Row:
        def __init__(self, concept_id, code_id, score):
            self.concept_id, self.code_id, self.score = concept_id, code_id, score

    selected = similarity._select_adaptive(
        [Row(1, 10, 0.10)], min_score=0.25, rel_margin=0.05, max_k=15
    )
    assert selected == {}
