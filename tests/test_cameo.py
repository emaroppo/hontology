"""CAMEO codebook parsing and loading."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from hontology.db.models import Code, CodeSystem
from hontology.db.session import session_scope
from hontology.ingest import cameo

SAMPLE = """CAMEOEVENTCODE\tEVENTDESCRIPTION
01\tMAKE PUBLIC STATEMENT
010\tMake statement, not specified below
011\tDecline comment
14\tPROTEST
145\tProtest violently, riot
1451\tEngage in political dissent, riot
"""


def test_parse_lookup_skips_the_header():
    rows = cameo.parse_lookup(SAMPLE)
    assert ("CAMEOEVENTCODE", "EVENTDESCRIPTION") not in rows
    assert len(rows) == 6


def test_parse_lookup_keeps_codes_as_strings():
    """Leading zeros are significant: root 01 is not the integer 1."""
    rows = dict(cameo.parse_lookup(SAMPLE))
    assert "01" in rows
    assert rows["01"] == "MAKE PUBLIC STATEMENT"


def test_parse_lookup_ignores_unknown_widths():
    rows = cameo.parse_lookup("1\tone digit\n12345\tfive digits\n14\tPROTEST\n")
    assert [c for c, _ in rows] == ["14"]


def test_resolve_chain_is_most_specific_first():
    assert cameo.resolve_chain("1451") == ["1451", "145", "14"]
    assert cameo.resolve_chain("145") == ["145", "14"]
    assert cameo.resolve_chain("14") == ["14"]


@pytest.mark.requires_db
class TestLoad:
    def test_load_assigns_levels_by_width(self):
        with session_scope() as session:
            cameo.load_codes(session, cameo.parse_lookup(SAMPLE))
            codes = {c.code: c.level for c in session.scalars(select(Code))}
        assert codes["01"] == "root"
        assert codes["010"] == "base"
        assert codes["1451"] == "event"

    def test_load_wires_the_parent_hierarchy(self):
        with session_scope() as session:
            cameo.load_codes(session, cameo.parse_lookup(SAMPLE))
            by_code = {c.code: c for c in session.scalars(select(Code))}
            assert by_code["1451"].parent_code_id == by_code["145"].id
            assert by_code["145"].parent_code_id == by_code["14"].id
            # A root code has no parent.
            assert by_code["14"].parent_code_id is None

    def test_reloading_is_idempotent(self):
        with session_scope() as session:
            first = cameo.load_codes(session, cameo.parse_lookup(SAMPLE))
        with session_scope() as session:
            second = cameo.load_codes(session, cameo.parse_lookup(SAMPLE))

        assert first["inserted"] == 6
        assert second["inserted"] == 0
        assert second["total"] == 6

        with session_scope() as session:
            assert len(list(session.scalars(select(Code)))) == 6
            assert len(list(session.scalars(select(CodeSystem)))) >= 1

    def test_reload_updates_a_changed_description(self):
        with session_scope() as session:
            cameo.load_codes(session, cameo.parse_lookup(SAMPLE))

        revised = SAMPLE.replace("PROTEST", "PROTEST (revised)")
        with session_scope() as session:
            result = cameo.load_codes(session, cameo.parse_lookup(revised))
            assert result["updated"] == 1
            code = session.scalar(select(Code).where(Code.code == "14"))
            assert code.name == "PROTEST (revised)"
