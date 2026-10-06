"""Comparing arms against the baseline on both levels of the evaluation plan.

One report, every run side by side:

- **Event level:** the calendar, raw and verified by review, with judging cost.
- **Article level:** end-to-end and judge-only scores on the labelled sample,
  with intervals from resampling documents.
- **Per arm:** the paired F1 difference against the baseline (the pre-registered
  decision rule) and McNemar on the labelled pairs as a cross-check.

The labelled sample counts only documents labelled against every concept, and
only as a prefix of the frozen order: a document labelled out of turn is
reported, because the sample is a valid random sample only as a prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Run, Verdict
from hontology.evalkit import article, calendar
from hontology.evalkit.evaluate import label_map
from hontology.evalkit.metrics import mcnemar
from hontology.judge import prompts
from hontology.ontology import hierarchy


def labelled_sample(session: Session, ontology_id: int, manifest: dict) -> dict:
    """The manifest's documents a person has labelled on every concept.

    Returns the truth map over those documents and how the labelled set sits in
    the frozen order: the contiguous prefix, and any labelled out of turn.
    """
    # Documents are labelled on leaves; internal classes are derived from them.
    concept_ids = hierarchy.leaves(session, ontology_id)
    labels = label_map(session, ontology_id)
    per_document: dict[int, dict[int, bool]] = {}
    for (doc_id, concept_id), matched in labels.items():
        per_document.setdefault(doc_id, {})[concept_id] = matched

    order = [row["document_id"] for row in manifest["order"]]
    complete = {d for d in order if set(per_document.get(d, {})) >= concept_ids}
    prefix = 0
    while prefix < len(order) and order[prefix] in complete:
        prefix += 1
    out_of_turn = sorted(complete - set(order[:prefix]))
    truth = {
        (doc_id, concept_id): per_document[doc_id][concept_id]
        for doc_id in order[:prefix]
        for concept_id in concept_ids
    }
    return {"truth": truth, "prefix": prefix, "out_of_turn": out_of_turn}


def compare_arms(
    session: Session,
    baseline_id: int,
    arm_ids: list[int],
    entries: list[calendar.Entry],
    manifest: dict | None,
    *,
    before: int = calendar.DEFAULT_BEFORE,
    after: int = calendar.DEFAULT_AFTER,
    seed: int = 0,
) -> dict:
    baseline = session.get(Run, baseline_id)
    if baseline is None:
        raise LookupError(f"run {baseline_id} does not exist")
    run_ids = [baseline_id, *arm_ids]

    report: dict = {"baseline": baseline_id, "arms": arm_ids, "runs": {}, "comparisons": {}}
    for run_id in run_ids:
        result = calendar.evaluate(session, run_id, entries, before=before, after=after)
        report["runs"][run_id] = {
            "calendar": result["summary"],
            "lead_times": result["lead_times"],
            "cost": result["cost"],
            "not_processed": result["not_processed"],
        }

    if manifest is None:
        return report

    sample = labelled_sample(session, baseline.ontology_id, manifest)
    truth = sample["truth"]
    report["sample"] = {
        "manifest_sha256": manifest.get("order_sha256"),
        "labelled_prefix": sample["prefix"],
        "out_of_turn": sample["out_of_turn"],
    }
    if not truth:
        return report

    keys = set(truth)
    predicted = {run_id: article.predictions(session, run_id, keys) for run_id in run_ids}
    for run_id in run_ids:
        judged = article.judged_keys(session, run_id, keys)
        report["runs"][run_id]["article"] = {
            "end_to_end": article.document_bootstrap(truth, predicted[run_id], seed=seed),
            "judge_only": article.document_bootstrap(
                {k: truth[k] for k in judged}, predicted[run_id], seed=seed
            ),
            "pairs_judged": len(judged),
        }
    report["sample"]["status"] = article.sample_status(
        report["runs"][baseline_id]["article"]["end_to_end"]
    )
    for arm_id in arm_ids:
        paired = mcnemar(truth, predicted[baseline_id], predicted[arm_id])
        report["comparisons"][arm_id] = {
            "f1": article.paired_document_bootstrap(
                truth, predicted[baseline_id], predicted[arm_id], seed=seed
            ),
            "mcnemar": paired.as_dict(),
        }
        arm = session.get(Run, arm_id)
        if arm is not None and _is_hierarchical(arm):
            report["comparisons"][arm_id]["hierarchy"] = hierarchy_diagnostics(
                session, arm_id, baseline_id, baseline.ontology_id, truth
            )
    return report


def hierarchy_diagnostics(
    session: Session,
    run_id: int,
    baseline_id: int,
    ontology_id: int,
    truth: dict[tuple[int, int], bool],
) -> dict:
    """Where a hierarchical run gains and loses, on the labelled sample.

    Internal classes are never labelled; an internal class is positive on a
    document exactly when one of the leaves beneath it is, which is what its
    union-worded definition says. From that:

    - **recall per level**, each level conditional on a parent having been
      answered yes, so a miss is located at the level where it happened;
    - **internal false positives**, kept apart because a class answered yes
      with no labelled leaf beneath it may be a gap in the ontology rather
      than a judge error;
    - **coverage**, the run's correct leaf matches split by whether the
      baseline's retrieval had selected that pair. A gain from reaching
      leaves retrieval missed is coverage, not structure.
    """
    parent_map = hierarchy.parents(session, ontology_id)
    child_map = hierarchy.children(session, ontology_id)
    leaves = hierarchy.leaves(session, ontology_id)
    internal = set(child_map)
    documents = {doc_id for doc_id, _ in truth}

    def below(class_id: int) -> set[int]:
        found: set[int] = set()
        stack = list(child_map.get(class_id, ()))
        while stack:
            current = stack.pop()
            if current not in found:
                found.add(current)
                stack.extend(child_map.get(current, ()))
        return found & leaves

    full_truth = dict(truth)
    for class_id in internal:
        under = below(class_id)
        for doc_id in documents:
            full_truth[(doc_id, class_id)] = any(
                truth.get((doc_id, leaf), False) for leaf in under
            )

    answered = {
        (doc_id, concept_id): bool(matched)
        for doc_id, concept_id, matched in session.execute(
            select(Verdict.document_id, Verdict.concept_id, Verdict.matched).where(
                Verdict.run_id == run_id,
                Verdict.document_id.in_(documents),
                Verdict.error.is_(None),
                Verdict.matched.is_not(None),
            )
        )
    }

    levels: dict[int, dict[str, int]] = {}
    for (doc_id, class_id), expected in full_truth.items():
        if not expected:
            continue
        parents = parent_map.get(class_id, set())
        if parents and not any(answered.get((doc_id, p), False) for p in parents):
            continue  # never reached: the miss belongs to a level above
        level = levels.setdefault(hierarchy.depth(parent_map, class_id), {"n": 0, "hits": 0})
        level["n"] += 1
        level["hits"] += int(answered.get((doc_id, class_id), False))

    internal_answers = [
        (key, yes) for key, yes in answered.items() if key[1] in internal and yes
    ]
    selected = set(
        session.execute(
            select(Candidate.document_id, Candidate.concept_id).where(
                Candidate.run_id == baseline_id,
                Candidate.selected.is_(True),
                Candidate.document_id.in_(documents),
            )
        ).all()
    )
    true_leaf_hits = [
        key for key, expected in truth.items() if expected and answered.get(key, False)
    ]
    return {
        "recall_by_level": {
            level: counts | {"recall": counts["hits"] / counts["n"] if counts["n"] else None}
            for level, counts in sorted(levels.items())
        },
        "internal_answered_yes": len(internal_answers),
        "internal_false_positives": sum(
            1 for key, _ in internal_answers if not full_truth[key]
        ),
        "coverage": {
            "true_positives": len(true_leaf_hits),
            "baseline_had_selected": sum(1 for key in true_leaf_hits if key in selected),
            "beyond_baseline_retrieval": sum(
                1 for key in true_leaf_hits if key not in selected
            ),
        },
    }


def _is_hierarchical(run: Run) -> bool:
    prompt_id = (run.config or {}).get("judge", {}).get("prompt_id")
    return bool(prompt_id) and prompts.get(prompt_id).mode == prompts.HIERARCHICAL


def load_manifest(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None
