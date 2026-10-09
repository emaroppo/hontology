"""A judge version scored on what it was given, across the runs that used it.

End-to-end recall mixes two failures: retrieval never offered the judge a true
match, or the judge saw it and said no. Only the second is the judge's, so a
judge is scored on the pairs it was responsible for:

- **Per-pair and batched prompts** answer exactly the pairs retrieval selected,
  so a judge is responsible for the labelled pairs it returned a verdict on.
- **Top-down prompts** (hierarchical, extraction) choose for themselves which
  leaves to descend to, and write no row for a leaf they never reached. For
  them, every labelled leaf of every article they processed counts, an
  unreached leaf as "no"; otherwise a miss at the top of the hierarchy would
  vanish from their recall.

Those two are not comparable across prompt styles: a top-down judge answers for
true matches retrieval never offered. The **retrieved** scope scores every judge
on the same footing instead: only the pairs retrieval selected (in the run whose
candidates it used), and of those only the ones it answered or, top-down, the
ones on articles it processed. Judges sharing a retrieval version are then
compared on the same pairs.

Runs with the same judge version are pooled; a pair judged by several of them
is counted once, from the newest run.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Run, Verdict
from hontology.evaluation.article.bootstrap import document_bootstrap
from hontology.evaluation.metrics.calibration import calibration_bins
from hontology.ontology import hierarchy
from hontology.pipeline.judge import prompts
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs import versions

Key = tuple[int, int]


def list_versions(session: Session, ontology_id: int) -> list[dict]:
    """Judge versions of an ontology's runs that judged anything, newest first."""
    judged_runs = set(
        session.scalars(
            select(Verdict.run_id).distinct().join(Run).where(Run.ontology_id == ontology_id)
        )
    )
    grouped: dict[str, dict] = {}
    for run in session.scalars(
        select(Run).where(Run.id.in_(judged_runs)).order_by(Run.id.desc())
    ):
        version = versions.judge_version(session, run)
        entry = grouped.setdefault(version["version"], version | {"runs": []})
        entry["runs"].append({"id": run.id, "name": run.name, "status": run.status})
    return list(grouped.values())


def _given(
    session: Session, run: Run, truth: dict[Key, bool], leaves: set[int], scope: str
) -> tuple[dict[Key, bool], dict[Key, float]]:
    """The labelled pairs *run*'s judge is scored on, with its answers, and the
    confidence it stated where it actually answered."""
    documents = {doc for doc, _ in truth}
    verdicts = {
        (v.document_id, v.concept_id): v
        for v in session.scalars(
            select(Verdict).where(
                Verdict.run_id == run.id, Verdict.document_id.in_(sorted(documents))
            )
        )
    }
    mode = prompts.get(run_config.normalize(run.config or {})["judge"]["prompt_id"]).mode
    answer = {
        key: bool(v.matched)
        for key, v in verdicts.items()
        if v.error is None and v.matched is not None
    }
    if mode in prompts.TOP_DOWN_MODES:
        processed = {doc for doc, _ in verdicts}
        given = {
            key: answer.get(key, False)
            for key in truth
            if key[0] in processed and key[1] in leaves
        }
    else:
        given = {key: answer[key] for key in truth if key in answer}
    if scope == "retrieved":
        source = versions.source_run(session, run)
        selected = set(
            session.execute(
                select(Candidate.document_id, Candidate.concept_id).where(
                    Candidate.run_id == source.id,
                    Candidate.selected.is_(True),
                    Candidate.document_id.in_(sorted(documents)),
                )
            ).tuples()
        )
        given = {key: matched for key, matched in given.items() if key in selected}
    confidence = {
        key: stated
        for key in given
        if key in answer and (stated := verdicts[key].confidence) is not None
    }
    return given, confidence


def evaluate(
    session: Session,
    run_ids: list[int],
    truth: dict[Key, bool],
    *,
    scope: str = "responsible",
    seed: int = 0,
) -> dict:
    """Per-run and pooled precision and recall on the pairs each judge was given."""
    runs = [
        run for run_id in sorted(run_ids, reverse=True) if (run := session.get(Run, run_id))
    ]
    if not runs:
        raise LookupError("no such runs")
    leaves = hierarchy.leaves(session, runs[0].ontology_id)
    truth = {key: matched for key, matched in truth.items() if key[1] in leaves}

    per_run = []
    pooled: dict[Key, bool] = {}
    stated: dict[Key, float] = {}
    for run in runs:  # newest first, so a pair keeps the newest run's answer
        given, confidence = _given(session, run, truth, leaves, scope)
        scores = document_bootstrap({k: truth[k] for k in given}, given, seed=seed)
        per_run.append({"run_id": run.id, "name": run.name, "pairs": len(given), **scores})
        for key, matched in given.items():
            if key not in pooled:
                pooled[key] = matched
                if key in confidence:
                    stated[key] = confidence[key]
    return {
        "scope": scope,
        "truth_pairs": len(truth),
        "pooled": {
            "pairs": len(pooled),
            **document_bootstrap({k: truth[k] for k in pooled}, pooled, seed=seed),
        },
        "runs": per_run,
        # Is the stated confidence a probability? Only pairs actually answered.
        "calibration": calibration_bins(
            [(stated[key], pooled[key] == truth[key]) for key in stated]
        ),
    }
