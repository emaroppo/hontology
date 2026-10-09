"""Crediting filter links one by one.

A link admits an article exactly when the filter would: through the most
specific code of one of its events, or one of its themes. Each link's counts
must agree with that rule, and "unique" must mean no other link admits it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hontology.db.models import Document, FeedArticle, FeedEvent, FeedSlice
from hontology.db.session import session_scope
from hontology.evaluation.stages import filter_eval
from hontology.ontology import service
from hontology.pipeline.ingest.articles import filter as ingest_filter

pytestmark = pytest.mark.requires_db


@pytest.fixture
def world():
    """Riot is linked to CAMEO 145 and the theme PROTEST; Flood to theme FLOOD.

    Articles: a (event 145) riot, b (event 1451, base 145) riot, c (theme
    PROTEST and FLOOD) not a riot but a flood, d (event 145, theme PROTEST) riot.
    """
    with session_scope() as session:
        ontology = service.create_ontology(session, slug="test-filter-eval", name="Fe")
        riot = service.create_concept(session, ontology.id, name="Riot")
        flood = service.create_concept(session, ontology.id, name="Flood")
        feed_slice = FeedSlice(
            feed="test_fe", slice_key="20261008000000", sliced_at=datetime.now(UTC), status="ok"
        )
        session.add(feed_slice)
        session.flush()
        docs = {}
        for key in "abcd":
            document = Document(url=f"https://fe.test/{key}", url_hash=f"fetest00000{key}")
            session.add(document)
            session.flush()
            docs[key] = document.id

        def event(doc: str, event_code=None, base=None):
            session.add(
                FeedEvent(
                    slice_id=feed_slice.id,
                    feed_event_id=f"{doc}-{event_code}-{base}",
                    document_id=docs[doc],
                    event_code=event_code,
                    base_code=base,
                    root_code="14",
                )
            )

        def article(doc: str, themes: list[str]):
            session.add(
                FeedArticle(
                    slice_id=feed_slice.id,
                    record_id=f"r{doc}",
                    document_id=docs[doc],
                    themes=themes,
                )
            )

        event("a", base="145")
        event("b", event_code="1451", base="145")  # the event tier decides: not 145
        article("c", ["PROTEST", "FLOOD"])
        event("d", base="145")
        article("d", ["PROTEST"])
        session.flush()
        links = [
            [riot.id, "cameo", "145"],
            [riot.id, "gkg-themes", "PROTEST"],
            [flood.id, "gkg-themes", "FLOOD"],
        ]
        truth = {
            (docs["a"], riot.id): True,
            (docs["b"], riot.id): True,
            (docs["c"], riot.id): False,
            (docs["d"], riot.id): True,
            (docs["c"], flood.id): True,
            (docs["a"], flood.id): False,
        }
        names = {riot.id: "Riot", flood.id: "Flood"}
        return {"docs": docs, "links": links, "truth": truth, "names": names}


def test_keys_follow_the_filters_own_rule(world):
    docs = world["docs"]
    with session_scope() as session:
        keys = filter_eval.document_keys(session, sorted(docs.values()), world["links"])
        # The same rule the filter applies, event by event.
        assert (
            ingest_filter.resolve_event(
                FeedEvent(event_code="1451", base_code="145", root_code="14"), {"145": {1}}
            )
            is None
        )
    assert keys[docs["a"]] == {("cameo", "145")}
    assert docs["b"] not in keys
    assert keys[docs["c"]] == {("gkg-themes", "PROTEST"), ("gkg-themes", "FLOOD")}
    assert keys[docs["d"]] == {("cameo", "145"), ("gkg-themes", "PROTEST")}


def test_each_link_is_credited_for_its_own_class(world):
    with session_scope() as session:
        report = filter_eval.labelled_report(
            session, world["links"], world["truth"], world["names"]
        )
    rows = {(r["system"], r["code"]): r for r in report["links"]}
    cameo = rows[("cameo", "145")]
    # a and d admitted and true, b missed; unique only for a (d is also PROTEST).
    assert (cameo["tp"], cameo["fp"], cameo["fn"], cameo["unique_tp"]) == (2, 0, 1, 1)
    protest = rows[("gkg-themes", "PROTEST")]
    # d true, c admitted but not a riot, a and b missed.
    assert (protest["tp"], protest["fp"], protest["fn"]) == (1, 1, 2)
    flood = rows[("gkg-themes", "FLOOD")]
    # c's flood is admitted, but so is c through PROTEST: not unique.
    assert (flood["tp"], flood["tn"], flood["unique_tp"]) == (1, 1, 0)
    assert (report["positives"], report["positives_admitted"]) == (4, 3)


def test_corpus_cost_counts_what_only_one_code_admits(world):
    with session_scope() as session:
        cost = filter_eval.corpus_cost(session, world["links"])["codes"]
    assert cost["cameo:145"]["admitted"] == 2  # a and d
    assert cost["cameo:145"]["only_this"] == 1  # a
    assert cost["gkg-themes:PROTEST"] == {"admitted": 2, "only_this": 0, "downloaded": 0}
    assert cost["gkg-themes:FLOOD"]["only_this"] == 0
