"""Near-duplicate articles: one representative per group of copies.

Wire stories are republished under many URLs. In a random 15-minute slice of
the knowledge graph, 23% of articles repeat another's headline, and an event
window is worse: every outlet runs the same agency copy of a typhoon. Judging
forty copies of one text costs forty calls and buys nothing, and in pair-level
metrics forty copies would count as forty independent data points.

So after fetching, documents whose bodies are near-identical are grouped, and
every copy points at one **representative** through ``duplicate_of``. Only
representatives go to retrieval and the judge; a copy inherits its
representative's verdicts wherever results are read.

**Method.** Bodies are normalised and cut into overlapping five-word shingles;
a MinHash signature estimates the Jaccard similarity of two shingle sets, and
locality-sensitive hashing over bands of the signature finds candidate pairs
without comparing every pair. A candidate is accepted only when its estimated
similarity reaches the threshold, so LSH decides what to compare, never what
counts as a duplicate.

**Stability.** A group's representative is its earliest document (lowest id),
and a document already pointing at a representative is never moved. Re-running
over a larger set can add copies to existing groups but never reshuffles which
document carries the verdicts.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.base import among
from hontology.db.models import Document

log = logging.getLogger(__name__)

SHINGLE_WORDS = 5
PERMUTATIONS = 64
BANDS = 16  # 16 bands of 4 rows: pairs above ~0.5 Jaccard become candidates.
ROWS = PERMUTATIONS // BANDS
DEFAULT_THRESHOLD = 0.8
# Too little text to fingerprint meaningfully; such documents stay unique.
MIN_SHINGLES = 20

_PRIME = (1 << 31) - 1
_rng = np.random.default_rng(20251005)
_A = _rng.integers(1, _PRIME, size=PERMUTATIONS, dtype=np.int64)
_B = _rng.integers(0, _PRIME, size=PERMUTATIONS, dtype=np.int64)
_WORD = re.compile(r"\w+")


def shingles(text: str) -> set[int]:
    """31-bit hashes of every five-word window of the lowercased text."""
    words = _WORD.findall(text.lower())
    if len(words) < SHINGLE_WORDS:
        return set()
    out: set[int] = set()
    for i in range(len(words) - SHINGLE_WORDS + 1):
        digest = hashlib.blake2b(
            " ".join(words[i : i + SHINGLE_WORDS]).encode(), digest_size=4
        ).digest()
        out.add(int.from_bytes(digest, "big") & _PRIME)
    return out


def signature(hashes: set[int]) -> np.ndarray:
    """The MinHash signature: per permutation, the minimum permuted hash."""
    values = np.fromiter(hashes, dtype=np.int64, count=len(hashes))
    permuted = (_A[:, None] * values[None, :] + _B[:, None]) % _PRIME
    return permuted.min(axis=1)


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Estimated Jaccard similarity: the share of agreeing signature slots."""
    return float(np.mean(left == right))


def clusters(
    signatures: dict[int, np.ndarray], *, threshold: float = DEFAULT_THRESHOLD
) -> list[set[int]]:
    """Groups of two or more documents whose bodies are near-identical.

    Union-find over accepted pairs, so similarity is transitive within a group.
    """
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for doc_id, sig in signatures.items():
        for band in range(BANDS):
            buckets[(band, sig[band * ROWS : (band + 1) * ROWS].tobytes())].append(doc_id)

    parent = {doc_id: doc_id for doc_id in signatures}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in checked:
                    continue
                checked.add(pair)
                if similarity(signatures[left], signatures[right]) >= threshold:
                    parent[find(left)] = find(right)

    found: dict[int, set[int]] = defaultdict(set)
    for doc_id in signatures:
        found[find(doc_id)].add(doc_id)
    return [members for members in found.values() if len(members) > 1]


def _body(document: Document) -> str | None:
    path = get_settings().scrape_cache_dir / (document.body_path or "")
    if not document.body_path or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def deduplicate(
    session: Session, document_ids: list[int], *, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Group the fetched documents among *document_ids* and record representatives.

    Documents already marked as copies keep their representative, and their
    representatives join the comparison so a new copy of an old story attaches
    to the same group.
    """
    documents = list(
        session.scalars(
            select(Document).where(
                among(Document.id, document_ids),
                Document.body_path.is_not(None),
                Document.is_junk.is_(False),
            )
        )
    )
    existing = {d.id: d.duplicate_of for d in documents if d.duplicate_of is not None}
    anchors = set(existing.values()) - {d.id for d in documents}
    if anchors:
        documents += list(session.scalars(select(Document).where(among(Document.id, anchors))))

    signatures: dict[int, np.ndarray] = {}
    too_short = 0
    for document in documents:
        body = _body(document)
        hashes = shingles(body) if body else set()
        if len(hashes) < MIN_SHINGLES:
            too_short += 1
            continue
        signatures[document.id] = signature(hashes)

    by_id = {d.id: d for d in documents}
    marked = 0
    for members in clusters(signatures, threshold=threshold):
        # A group that already has a representative keeps it, so verdicts never
        # move and no copy ever points at another copy. Only a new group takes
        # its earliest document.
        settled = sorted(
            {by_id[m].duplicate_of or m for m in members if by_id[m].duplicate_of}
            | {m for m in members if m in anchors}
        )
        target = settled[0] if settled else min(members)
        for doc_id in members:
            document = by_id[doc_id]
            if doc_id == target or document.duplicate_of is not None:
                continue
            if doc_id in anchors or doc_id in existing.values():
                # Another group's representative: leave it, rather than chain.
                continue
            document.duplicate_of = target
            marked += 1
    session.flush()

    representatives = {
        d.id for d in documents if d.duplicate_of is None and d.id in set(document_ids)
    }
    result = {
        "documents": len([d for d in documents if d.id in set(document_ids)]),
        "fingerprinted": len(signatures),
        "too_short": too_short,
        "newly_marked": marked,
        "representatives": len(representatives),
    }
    log.info("dedup: %s", result)
    return result


def representative_of(session: Session, document_ids: list[int]) -> dict[int, int]:
    """``{document id: the id whose verdicts it reads}`` — itself unless a copy."""
    return {
        doc_id: duplicate_of or doc_id
        for doc_id, duplicate_of in session.execute(
            select(Document.id, Document.duplicate_of).where(among(Document.id, document_ids))
        )
    }
