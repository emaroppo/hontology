"""A set of filter links scored three ways, link by link.

The pre-download filter fetches an article when one of its feed codes is linked
to a class. Which codes an article carries for this purpose does not depend on
the other links: for each feed event only its most specific code counts (the
first of event, base and root that is present, matched or not, as
`pipeline.ingest.articles.filter.resolve_event` decides), and every theme of its knowledge-graph
record counts. So each link can be credited on its own:

- **Against the labels**, for the class a link reaches: true positives (true
  matches it admits), false positives (admitted, not a match: downloads that
  bought nothing), false negatives (true matches it does not admit) and true
  negatives. And its *unique* true positives: true matches no other link
  admits, so removing it loses exactly these. Labelled articles were mostly
  downloaded because the filter admitted them, so these counts flatter the
  filter, its false negatives most of all.
- **Against the calendar**, which is known independently of the filter: of each
  kind of event, how many have an article in their window that the links of the
  event's own class admit, and per code how many events it reaches that way.
- **Cost over the whole corpus**: per code, the feed articles it admits and those
  only it admits, which removing it would stop downloading. This walks every
  feed record, so it is computed on request.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.orm import Session

from hontology.evaluation.calendar import events
from hontology.pipeline.ingest.codes.cameo import CAMEO_SLUG
from hontology.pipeline.ingest.codes.themes import THEMES_SLUG
from hontology.pipeline.runs import versions

Key = tuple[str, str]  # (system slug, code)

# For each event, the code the filter checks: its most specific one present.
_EVENT_CODE = "COALESCE(NULLIF(event_code, ''), NULLIF(base_code, ''), NULLIF(root_code, ''))"


def link_set(session: Session, ontology_id: int, version: str | None) -> list[list]:
    """``[[concept id, system, code], ...]`` of a snapshot, or the live links."""
    if version is None:
        return versions.current_links(session, ontology_id)
    links = versions.link_snapshot(session, ontology_id, version)
    if links is None:
        raise LookupError(f"no link version {version!r}")
    return links


def _codes(links: list[list], system: str) -> list[str]:
    return sorted({code for _, s, code in links if s == system})


def document_keys(
    session: Session, document_ids: list[int], links: list[list]
) -> dict[int, set[Key]]:
    """The linked codes each document carries, as the filter reads them."""
    out: dict[int, set[Key]] = defaultdict(set)
    if not document_ids:
        return out
    cameo, themes = _codes(links, CAMEO_SLUG), _codes(links, THEMES_SLUG)
    if cameo:
        for doc, code in session.execute(
            text(
                f"SELECT DISTINCT document_id, {_EVENT_CODE} FROM feed_events "
                f"WHERE document_id = ANY(:docs) AND {_EVENT_CODE} = ANY(:codes)"
            ),
            {"docs": document_ids, "codes": cameo},
        ):
            out[doc].add((CAMEO_SLUG, code))
    if themes:
        for doc, theme in session.execute(
            text(
                "SELECT DISTINCT document_id, theme FROM feed_articles, unnest(themes) theme "
                "WHERE document_id = ANY(:docs) AND themes && CAST(:themes AS varchar[]) "
                "AND theme = ANY(:themes)"
            ),
            {"docs": document_ids, "themes": themes},
        ):
            out[doc].add((THEMES_SLUG, theme))
    return out


def _link_rows(links: list[list], names: dict[int, str]) -> list[dict]:
    return [
        {"class": names.get(cid, str(cid)), "concept_id": cid, "system": system, "code": code}
        for cid, system, code in links
    ]


def labelled_report(
    session: Session,
    links: list[list],
    truth: dict[tuple[int, int], bool],
    names: dict[int, str],
) -> dict:
    """Per link, its confusion against the labels for the class it reaches."""
    documents = sorted({doc for doc, _ in truth})
    keys = document_keys(session, documents, links)
    admitted = {doc for doc, found in keys.items() if found}
    rows = []
    for row in _link_rows(links, names):
        key = (row["system"], row["code"])
        tp = fp = fn = tn = unique_tp = 0
        for (doc, cid), matched in truth.items():
            if cid != row["concept_id"]:
                continue
            hit = key in keys.get(doc, ())
            if matched and hit:
                tp += 1
                unique_tp += keys[doc] == {key}
            elif matched:
                fn += 1
            elif hit:
                fp += 1
            else:
                tn += 1
        rows.append(row | {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "unique_tp": unique_tp})
    positives = [pair for pair, matched in truth.items() if matched]
    found = sum(1 for doc, _ in positives if doc in admitted)
    return {
        "documents": len(documents),
        "documents_admitted": len(admitted),
        "positives": len(positives),
        "positives_admitted": found,
        "links": rows,
    }


def calendar_report(
    session: Session,
    links: list[list],
    entries: list,
    names: dict[int, str],
    *,
    before: int = 1,
    after: int = 2,
) -> dict:
    """Per kind of calendar event, how many the links reach, and per code.

    Two counts, because the loose one is nearly always met: a country's window
    holds hundreds of articles, and some link admits one of them. The strict one
    asks whether the links of the event's *own* class admit an article in its
    window, which is what the filter is for.
    """
    ids = {name: cid for cid, name in names.items()}
    own_keys: dict[int, set[Key]] = defaultdict(set)
    for cid, system, code in links:
        own_keys[cid].add((system, code))
    loci = events.loci_for(session, entries)
    windows = {
        entry.id: set(events.window_documents(session, entry, loci, before=before, after=after))
        for entry in entries
    }
    keys = document_keys(session, sorted(set().union(*windows.values())), links)
    by_kind: dict[str, dict] = {}
    per_key: dict[Key, set[str]] = defaultdict(set)
    lost = []
    for entry in entries:
        reached = {key for doc in windows[entry.id] for key in keys.get(doc, ())}
        own = reached & own_keys.get(ids.get(entry.concept, -1), set())
        counts = by_kind.setdefault(
            entry.kind, {"events": 0, "observable": 0, "reached": 0, "reached_by_own_class": 0}
        )
        counts["events"] += 1
        counts["observable"] += bool(windows[entry.id])
        counts["reached"] += bool(reached)
        counts["reached_by_own_class"] += bool(own)
        for key in own:
            per_key[key].add(entry.id)
        if windows[entry.id] and not own:
            lost.append(
                {
                    "id": entry.id,
                    "kind": entry.kind,
                    "class": entry.concept,
                    "known_class": entry.concept in ids,
                    "description": entry.description,
                }
            )
    return {
        "by_kind": by_kind,
        # Events each code reaches through the class it is linked to.
        "per_code": {f"{s}:{c}": sorted(e) for (s, c), e in per_key.items()},
        "not_reached_by_own_class": lost,
    }


_CORPUS_COST = text(
    f"""
    WITH keys AS (
        SELECT DISTINCT document_id, 'cameo' AS system, {_EVENT_CODE} AS code
        FROM feed_events
        WHERE document_id IS NOT NULL AND {_EVENT_CODE} = ANY(:cameo)
        UNION
        SELECT DISTINCT document_id, 'gkg-themes', theme
        FROM feed_articles, unnest(themes) theme
        WHERE document_id IS NOT NULL AND themes && CAST(:themes AS varchar[])
          AND theme = ANY(:themes)
    ),
    per_document AS (SELECT document_id, COUNT(*) AS n FROM keys GROUP BY document_id)
    SELECT k.system, k.code,
           COUNT(*) AS admitted,
           COUNT(*) FILTER (WHERE p.n = 1) AS only_this,
           COUNT(*) FILTER (WHERE d.fetched_at IS NOT NULL) AS downloaded
    FROM keys k
    JOIN per_document p USING (document_id)
    JOIN documents d ON d.id = k.document_id
    GROUP BY k.system, k.code
    """
)


def corpus_cost(session: Session, links: list[list]) -> dict:
    """Per code, feed articles it admits and those only it admits."""
    rows = {
        f"{system}:{code}": {"admitted": admitted, "only_this": only, "downloaded": downloaded}
        for system, code, admitted, only, downloaded in session.execute(
            _CORPUS_COST,
            {"cameo": _codes(links, CAMEO_SLUG), "themes": _codes(links, THEMES_SLUG)},
        )
    }
    return {"codes": rows}
