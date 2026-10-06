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

from hontology.db.models import Concept, Run
from hontology.evalkit import article, calendar
from hontology.evalkit.evaluate import label_map
from hontology.evalkit.metrics import mcnemar


def labelled_sample(session: Session, ontology_id: int, manifest: dict) -> dict:
    """The manifest's documents a person has labelled on every concept.

    Returns the truth map over those documents and how the labelled set sits in
    the frozen order: the contiguous prefix, and any labelled out of turn.
    """
    concept_ids = set(
        session.scalars(select(Concept.id).where(Concept.ontology_id == ontology_id))
    )
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
    return report


def load_manifest(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None
