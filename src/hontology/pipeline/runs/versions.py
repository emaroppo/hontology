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

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import (
    OntologySnapshot,
    Run,
)
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs.config import stable_hash
from hontology.pipeline.runs.fingerprints import prompt_fingerprint, retrieval_code_hash

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
        "version": "r-" + stable_hash([settings, leaves, code], 10),
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
        + stable_hash(
            [
                judge,
                config["common"]["judge_body_limit"],
                snapshot.content_hash if snapshot else run.ontology_version,
                prompt,
            ],
            10,
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
