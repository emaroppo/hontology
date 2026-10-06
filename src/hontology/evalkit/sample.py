"""Drawing the labelled document sample, in an order frozen before labelling.

Article-level precision and recall need documents a person has labelled against
every concept, drawn independently of whichever arm will be scored on them. The
frame is the representative documents the calendar's windows actually fetched,
including those retrieval found nothing in: leaving those out would hide every
concept the system never surfaced, which is exactly what recall must count.

**One frozen order, labelled as a prefix.** Labelling stops when the baseline's
intervals are narrow enough, so the sample size is not known in advance. The
order is therefore built so that *any prefix* is a proportional, stratified
random sample: within each stratum documents are shuffled, and each gets a key
``(position + jitter) / stratum size`` that spreads it evenly through the list.
Stopping after 120 documents, or 470, gives the same design either way.

**Strata** are the window a document first appears in, crossed with how well
retrieval scored it (no candidate, low, middle, high), so a prefix cannot fill up
with one busy window or with only easy, high-scoring articles.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Document
from hontology.evalkit.calendar import Entry, loci_for, window_documents
from hontology.ingest.dedup import representative_of

# Best retrieval score per document, banded. Fixed bands rather than quantiles,
# so the strata mean the same thing for every run they are drawn from.
SCORE_BANDS = ((0.6, "low"), (0.7, "middle"), (float("inf"), "high"))


@dataclass(frozen=True)
class SampledDocument:
    document_id: int
    url: str
    window: str
    band: str

    @property
    def stratum(self) -> str:
        return f"{self.window}/{self.band}"


def score_band(best: float | None) -> str:
    if best is None:
        return "none"
    for upper, name in SCORE_BANDS:
        if best < upper:
            return name
    return SCORE_BANDS[-1][1]


def frame(
    session: Session, run_id: int, entries: list[Entry], *, before: int, after: int
) -> list[SampledDocument]:
    """Every fetched representative in the entries' windows, with its stratum.

    A document seen in several windows belongs to the first entry, in calendar
    order, whose window contains it, so each document is in exactly one stratum.
    """
    loci = loci_for(session, entries)
    window_of: dict[int, str] = {}
    for entry in entries:
        docs = window_documents(session, entry, loci, before=before, after=after)
        for rep in set(representative_of(session, sorted(docs)).values()):
            window_of.setdefault(rep, entry.id)

    usable = {
        doc_id: url
        for doc_id, url in session.execute(
            select(Document.id, Document.url).where(
                Document.id.in_(window_of),
                Document.body_path.is_not(None),
                Document.is_junk.is_(False),
                Document.duplicate_of.is_(None),
            )
        )
    }
    best = {
        doc_id: float(score)
        for doc_id, score in session.execute(
            select(Candidate.document_id, func.max(Candidate.score))
            .where(Candidate.run_id == run_id, Candidate.document_id.in_(usable))
            .group_by(Candidate.document_id)
        )
    }
    return [
        SampledDocument(doc_id, url, window_of[doc_id], score_band(best.get(doc_id)))
        for doc_id, url in sorted(usable.items())
    ]


def frozen_order(documents: list[SampledDocument], seed: int) -> list[SampledDocument]:
    """An order in which every prefix is a proportional stratified sample."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[SampledDocument]] = defaultdict(list)
    for document in documents:
        by_stratum[document.stratum].append(document)

    keyed: list[tuple[float, str, SampledDocument]] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda d: d.document_id)
        rng.shuffle(members)
        size = len(members)
        for position, document in enumerate(members):
            keyed.append(((position + rng.random()) / size, stratum, document))
    keyed.sort(key=lambda item: (item[0], item[1], item[2].document_id))
    return [document for _, _, document in keyed]


def manifest(
    run_id: int, calendar_sha256: str, seed: int, ordered: list[SampledDocument]
) -> dict:
    """The frozen sample: everything needed to reproduce or audit it."""
    order = [
        {"document_id": d.document_id, "url": d.url, "stratum": d.stratum} for d in ordered
    ]
    strata: dict[str, int] = defaultdict(int)
    for d in ordered:
        strata[d.stratum] += 1
    body = {
        "run_id": run_id,
        "calendar_sha256": calendar_sha256,
        "seed": seed,
        "size": len(order),
        "strata": dict(sorted(strata.items())),
        "order": order,
    }
    canonical = json.dumps(order, sort_keys=True, separators=(",", ":"))
    return body | {"order_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
