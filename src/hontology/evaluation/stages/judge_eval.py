"""A judge version scored on what it was shown, across the runs that used it.

End-to-end recall mixes two failures: retrieval never showed the judge a true
match, or the judge was shown it and said no. Only the second is the judge's,
so a judge is scored on the labelled pairs it was shown (`evaluation.shown`),
in every prompt mode alike:

- **Per-pair and batched prompts** are shown the pairs their selection kept.
- **Top-down prompts** are shown the leaves routing or extraction offered. A
  leaf they never reached was not shown; it is retrieval's miss, not theirs.
- **A failed call counts as "no".** The pair was shown; liveness reports it.

Judges are then compared on one footing whatever their prompt mode, and two
runs sharing a retrieval key are compared on the very same pairs.

Runs with the same judge version are pooled; a pair judged by several of them
is counted once, from the newest run. A run without a shown record (made
before the record existed, and not yet backfilled) contributes no pairs, and
says so.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Run, Verdict
from hontology.evaluation import shown as shown_module
from hontology.evaluation.article.bootstrap import document_bootstrap
from hontology.evaluation.metrics.calibration import calibration_bins
from hontology.ontology import hierarchy
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
    session: Session, run: Run, truth: dict[Key, bool]
) -> tuple[dict[Key, bool], dict[Key, float]]:
    """The labelled pairs *run*'s judge was shown, with its answers, and the
    confidence it stated where it actually answered."""
    documents = {doc for doc, _ in truth}
    shown = shown_module.shown_pairs(session, run.id, documents) & set(truth)
    given = shown_module.judge_answers(session, run.id, shown)
    confidence = {
        (v.document_id, v.concept_id): stated
        for v in session.scalars(
            select(Verdict).where(
                Verdict.run_id == run.id,
                Verdict.error.is_(None),
                Verdict.matched.is_not(None),
                Verdict.document_id.in_(sorted(documents)),
            )
        )
        if (v.document_id, v.concept_id) in shown and (stated := v.confidence) is not None
    }
    return given, confidence


def evaluate(
    session: Session,
    run_ids: list[int],
    truth: dict[Key, bool],
    *,
    seed: int = 0,
) -> dict:
    """Per-run and pooled precision and recall on the pairs each judge was shown."""
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
        given, confidence = _given(session, run, truth)
        scores = document_bootstrap({k: truth[k] for k in given}, given, seed=seed)
        per_run.append(
            {
                "run_id": run.id,
                "name": run.name,
                "pairs": len(given),
                "shown_record": shown_module.has_record(session, run.id),
                **scores,
            }
        )
        for key, matched in given.items():
            if key not in pooled:
                pooled[key] = matched
                if key in confidence:
                    stated[key] = confidence[key]
    return {
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
