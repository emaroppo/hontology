"""Snapshot hashing rules.

These run without a database: `content_hash` is deliberately a pure function over
plain rows so the versioning rules are testable in isolation.
"""

from __future__ import annotations

from hontology.ontology.snapshots import content_hash, normalize_set


def concept(cid: int, **overrides) -> dict:
    row = {
        "id": cid,
        "name": f"Concept {cid}",
        "definition": "Something observable happened.",
        "inclusion_criteria": "It has already occurred.",
        "exclusion_criteria": "Planned or threatened occurrences.",
        "weight": 1.0,
        "category": "default",
    }
    row.update(overrides)
    return row


def test_identical_content_hashes_identically():
    a = [concept(1), concept(2)]
    b = [concept(1), concept(2)]
    assert content_hash(a) == content_hash(b)


def test_order_does_not_affect_the_hash():
    forward = [concept(1), concept(2), concept(3)]
    reversed_ = list(reversed(forward))
    assert content_hash(forward) == content_hash(reversed_)


def test_edge_whitespace_does_not_mint_a_version():
    plain = [concept(1, definition="A port closes.")]
    padded = [concept(1, definition="  A port closes.\n")]
    assert content_hash(plain) == content_hash(padded)


def test_changing_a_definition_changes_the_hash():
    before = [concept(1, definition="A port closes.")]
    after = [concept(1, definition="A port suspends vessel operations.")]
    assert content_hash(before) != content_hash(after)


def test_changing_criteria_changes_the_hash():
    """Criteria reach the judge prompt verbatim, so they are part of identity."""
    before = [concept(1, exclusion_criteria="Planned closures.")]
    after = [concept(1, exclusion_criteria="Planned or announced closures.")]
    assert content_hash(before) != content_hash(after)


def test_weight_does_not_change_the_hash():
    """The guarantee that lets a user retune scoring without rotting labels."""
    light = [concept(1, weight=0.1)]
    heavy = [concept(1, weight=99.0)]
    assert content_hash(light) == content_hash(heavy)


def test_category_does_not_change_the_hash():
    a = [concept(1, category="logistics")]
    b = [concept(1, category="security")]
    assert content_hash(a) == content_hash(b)


def test_adding_a_concept_changes_the_hash():
    assert content_hash([concept(1)]) != content_hash([concept(1), concept(2)])


def test_normalize_set_keeps_only_hashed_fields():
    (row,) = normalize_set([concept(1)])
    assert set(row) == {
        "id",
        "name",
        "definition",
        "inclusion_criteria",
        "exclusion_criteria",
    }


def test_none_and_empty_string_are_equivalent():
    """A field cleared in the UI arrives as either; they must not differ."""
    empty = [concept(1, inclusion_criteria="")]
    missing = [concept(1, inclusion_criteria=None)]
    assert content_hash(empty) == content_hash(missing)
