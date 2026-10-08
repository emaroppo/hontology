"""Per-stage versions: which filter, retrieval and judge a run used.

Each stage is versioned by what decides its output, so runs can be grouped by
stage and a stage scored across every run that used it:

- **Filter**: the set of code links, snapshotted when a run applies them
  (``f1``, ``f2``, ...). Links are live rows a tick changes, so without a
  snapshot nothing records which links a run fetched with.
- **Retrieval**: what decides the *ranking*: the embedding model, which class
  fields are embedded, the article text limit, the leaves' wording, and a hash
  of the ranking and selection code. The cutoff is deliberately not part of it:
  it is a parameter, tuned after the fact over a ranking that is cheap and
  deterministic to reproduce. A run that reused another's candidates has that
  run's retrieval.
- **Judge**: the judge settings, the article text limit, the ontology's content
  and a fingerprint of the prompt *text*, so a template edited in code is a new
  version even though its id did not change.

Code and prompt fingerprints are recorded on a run when it is created; a run
from before then is fingerprinted with the current code, which is right as long
as that code has not changed since.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    Code,
    CodeSystem,
    Concept,
    ConceptCode,
    Document,
    LinkSnapshot,
    OntologySnapshot,
    Run,
)
from hontology.evalkit import config as run_config


def _hash(payload: Any, n: int = 10) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Code and prompt fingerprints
# ---------------------------------------------------------------------------


@functools.cache
def retrieval_code_hash() -> str:
    """The code that turns embeddings into a ranking and a ranking into a cut."""
    from hontology.retrieve import candidates, embed, tuning

    parts = [
        inspect.getsource(candidates.select_adaptive),
        inspect.getsource(candidates.build_semantic),
        candidates._NEAREST_CONCEPTS.text,
        inspect.getsource(embed.concept_text),
        inspect.getsource(embed.normalize_for_embedding),
        inspect.getsource(embed._prefixes),
        tuning._POOL.text,
    ]
    return _hash(parts, 12)


@functools.cache
def prompt_fingerprint(prompt_id: str) -> str:
    """A hash of everything a template sends, rendered on fixed inputs.

    Rendering rather than hashing source catches an edit to a shared constant
    (a response shape, say) that a builder only refers to.
    """
    from hontology.judge import prompts

    template = prompts.get(prompt_id)
    document = Document(url="https://example.test/a", title="Title")
    concepts = [
        Concept(id=1, name="Alpha", definition="D1", inclusion_criteria="I1"),
        Concept(id=2, name="Beta", definition="D2", exclusion_criteria="E2"),
    ]
    event = {"description": "An event.", "country": "XX", "quote": "A quote."}
    parts: list[Any] = [template.prompt_id, template.mode]
    for name, value in vars(template).items():
        if isinstance(value, str | int) or value is None:
            parts.append((name, value))
            continue
        attempts = (
            lambda f: f(document, concepts[0], "BODY", 100),
            lambda f: f(document, concepts, "BODY", 100),
            lambda f: f(event, concepts),
            lambda f: f(event, concepts, True),
        )
        for attempt in attempts:
            try:
                parts.append((name, attempt(value)))
                break
            except Exception:  # noqa: BLE001, S112 - try the next call shape
                continue
        else:
            parts.append((name, inspect.getsource(value)))
    return _hash(parts, 12)


def recorded(run: Run) -> dict:
    """Fingerprints to stamp on a run at creation."""
    judge = run_config.normalize(run.config or {})["judge"]
    return {
        "retrieval_code": retrieval_code_hash(),
        "prompt": prompt_fingerprint(judge["prompt_id"]),
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def source_run(session: Session, run: Run) -> Run:
    """The run whose candidates *run* used: itself, unless it reused another's."""
    seen = {run.id}
    while True:
        manifest = run.manifest or {}
        source_id = (manifest.get("sample") or {}).get("candidates_from") or (
            manifest.get("documents") or {}
        ).get("candidates_from")
        if not source_id or source_id in seen:
            return run
        source = session.get(Run, int(source_id))
        if source is None:
            return run
        seen.add(source.id)
        run = source


def _snapshot(session: Session, run: Run) -> OntologySnapshot | None:
    return session.scalar(
        select(OntologySnapshot).where(
            OntologySnapshot.ontology_id == run.ontology_id,
            OntologySnapshot.version == run.ontology_version,
        )
    )


def leaf_wording(session: Session, run: Run) -> list[dict]:
    """The leaves as they were worded at the run's ontology version."""
    snapshot = _snapshot(session, run)
    if snapshot is None:
        return []
    concepts = json.loads(snapshot.payload)
    parents = {parent for _, parent in json.loads(snapshot.edges or "[]")}
    return [c for c in concepts if c["id"] not in parents]


def retrieval_version(session: Session, run: Run) -> dict:
    source = source_run(session, run)
    config = run_config.normalize(source.config or {})
    candidates = config["candidates"]
    settings = {
        "embed_provider": candidates["embed_provider"],
        "embed_model": candidates["embed_model"],
        "concept_fields": candidates["concept_fields"],
        "embed_body_limit": config["common"]["embed_body_limit"],
    }
    leaves = leaf_wording(session, source)
    code = (source.manifest or {}).get("versions", {}).get("retrieval_code")
    code = code or retrieval_code_hash()
    return {
        "version": "r-" + _hash([settings, leaves, code]),
        "settings": settings,
        "ontology_version": source.ontology_version,
        "leaves": len(leaves),
        "code": code,
        "source_run": source.id,
    }


def judge_version(session: Session, run: Run) -> dict:
    config = run_config.normalize(run.config or {})
    judge = config["judge"]
    snapshot = _snapshot(session, run)
    prompt = (run.manifest or {}).get("versions", {}).get("prompt")
    prompt = prompt or prompt_fingerprint(judge["prompt_id"])
    return {
        "version": "j-"
        + _hash(
            [
                judge,
                config["common"]["judge_body_limit"],
                snapshot.content_hash if snapshot else run.ontology_version,
                prompt,
            ]
        ),
        "provider": judge["provider"],
        "model": judge["model"],
        "prompt_id": judge["prompt_id"],
        "prompt": prompt,
        "ontology_version": run.ontology_version,
    }


def note_filter(run: Run, version: str) -> None:
    """Record that *run* fetched with link snapshot *version*."""
    recorded_versions = dict((run.manifest or {}).get("versions", {}))
    used = set(recorded_versions.get("filter", []))
    if version not in used:
        recorded_versions["filter"] = sorted(used | {version})
        run.manifest = (run.manifest or {}) | {"versions": recorded_versions}


def filter_versions(run: Run) -> list[str]:
    """The link snapshots a run fetched with; empty if it never applied the filter.
    Several when the links changed while it ran, window by window."""
    return list((run.manifest or {}).get("versions", {}).get("filter", []))


# ---------------------------------------------------------------------------
# Link snapshots
# ---------------------------------------------------------------------------


def current_links(session: Session, ontology_id: int) -> list[list]:
    """``[[concept id, system slug, code], ...]``, sorted."""
    rows = session.execute(
        select(ConceptCode.concept_id, CodeSystem.slug, Code.code)
        .join(Code, Code.id == ConceptCode.code_id)
        .join(CodeSystem, CodeSystem.id == Code.system_id)
        .join(Concept, Concept.id == ConceptCode.concept_id)
        .where(Concept.ontology_id == ontology_id)
    )
    return sorted([concept_id, system, code] for concept_id, system, code in rows)


def resolve_links(session: Session, ontology_id: int) -> LinkSnapshot | None:
    """The snapshot of the live links, minting one if they changed. None when
    there are no links, since an ontology without them has no filter."""
    links = current_links(session, ontology_id)
    if not links:
        return None
    digest = _hash(links, 64)
    existing = session.scalar(
        select(LinkSnapshot).where(
            LinkSnapshot.ontology_id == ontology_id, LinkSnapshot.content_hash == digest
        )
    )
    if existing is not None:
        return existing
    count = len(
        list(
            session.scalars(
                select(LinkSnapshot.id).where(LinkSnapshot.ontology_id == ontology_id)
            )
        )
    )
    snapshot = LinkSnapshot(
        ontology_id=ontology_id,
        version=f"f{count + 1}",
        content_hash=digest,
        n_links=len(links),
        payload=json.dumps(links),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def link_snapshot(session: Session, ontology_id: int, version: str) -> list[list] | None:
    snapshot = session.scalar(
        select(LinkSnapshot).where(
            LinkSnapshot.ontology_id == ontology_id, LinkSnapshot.version == version
        )
    )
    return json.loads(snapshot.payload) if snapshot else None
