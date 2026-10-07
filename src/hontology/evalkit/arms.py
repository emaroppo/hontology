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

from hontology.db.base import among
from hontology.db.models import Candidate, Run, Verdict
from hontology.evalkit import article, calendar
from hontology.evalkit.evaluate import label_map
from hontology.evalkit.metrics import mcnemar
from hontology.evalkit.sample import EQUAL_PER_WINDOW
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
        run = session.get(Run, run_id)
        report["runs"][run_id] = {
            "ontology_version": run.ontology_version if run else None,
            "prompt_id": ((run.config or {}).get("judge") or {}).get("prompt_id")
            if run
            else None,
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
    # An equal-per-window sample over-represents small windows; weighting each
    # document by its window's size keeps pooled scores an estimate of the frame.
    weights: dict[int, float] | None = None
    groups: dict[int, str] | None = None
    if manifest.get("allocation") == EQUAL_PER_WINDOW:
        weights, groups = article.window_weights(manifest, {doc_id for doc_id, _ in keys})
    report["sample"]["allocation"] = manifest.get("allocation", "proportional")
    predicted = {run_id: article.predictions(session, run_id, keys) for run_id in run_ids}
    for run_id in run_ids:
        judged = article.judged_keys(session, run_id, keys)
        report["runs"][run_id]["article"] = {
            "end_to_end": article.document_bootstrap(
                truth, predicted[run_id], seed=seed, weights=weights, groups=groups
            ),
            "judge_only": article.document_bootstrap(
                {k: truth[k] for k in judged},
                predicted[run_id],
                seed=seed,
                weights=weights,
                groups=groups,
            ),
            "pairs_judged": len(judged),
            # On the labelled documents only, so arms that judged different
            # amounts of the calendar are compared on the same articles.
            "cost_on_sample": calendar.judging_cost(
                session, run_id, {doc_id for doc_id, _ in keys}
            ),
        }
    report["sample"]["status"] = article.sample_status(
        report["runs"][baseline_id]["article"]["end_to_end"]
    )
    for arm_id in arm_ids:
        paired = mcnemar(truth, predicted[baseline_id], predicted[arm_id])
        report["comparisons"][arm_id] = {
            "f1": article.paired_document_bootstrap(
                truth,
                predicted[baseline_id],
                predicted[arm_id],
                seed=seed,
                weights=weights,
                groups=groups,
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
                among(Verdict.document_id, documents),
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
                among(Candidate.document_id, documents),
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
    return bool(prompt_id) and prompts.get(prompt_id).mode in prompts.TOP_DOWN_MODES


def load_manifest(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None


def _ci(value: float | None, ci: list[float | None] | tuple) -> str:
    if value is None:
        return "-"
    low, high = (list(ci) + [None, None])[:2]
    if low is None or high is None:
        return f"{value:.2f}"
    return f"{value:.2f} ({low:.2f}\u2013{high:.2f})"


def _rate_cell(stat: dict | None) -> str:
    if not stat or stat.get("n") in (None, 0):
        return "-"
    return f"{stat['hits']}/{stat['n']} = " + _ci(stat["rate"], stat.get("ci", [None, None]))


def render_markdown(report: dict) -> str:
    """The arms report as Markdown tables, for the results write-up.

    Every number carries its interval, and cost is shown beside the scores it
    bought. Article-level cost is on the labelled documents only, so an arm
    that judged more of the calendar does not look more expensive for it.
    """
    baseline = report["baseline"]
    runs = report["runs"]
    name = {
        run_id: f"{run_id} (baseline)" if run_id == baseline else str(run_id) for run_id in runs
    }
    out: list[str] = []
    out.append("| Run | Ontology version | Prompt |\n| --- | --- | --- |")
    for r in runs:
        version = runs[r].get("ontology_version") or "-"
        out.append(f"| {name[r]} | {version} | {runs[r].get('prompt_id') or '-'} |")

    sample = report.get("sample")
    if sample and any("article" in runs[r] for r in runs):
        status = sample.get("status") or {}
        out.append(
            f"\n## Article level\n\nLabelled sample: the first {sample['labelled_prefix']} "
            "documents of the frozen order "
            f"(manifest {str(sample.get('manifest_sha256'))[:12]}). "
            "Intervals are 95%, resampling whole documents. End to end, a pair never "
            "judged counts as no; judge only scores the pairs each run judged."
            + (
                " Every calendar window has an equal share of the sample, so each "
                "document is weighted by its window's size over the number labelled "
                "from it, and the bootstrap resamples within windows; the McNemar "
                "counts below are unweighted."
                if sample.get("allocation") == EQUAL_PER_WINDOW
                else ""
            )
        )
        out.append(
            "\n| Run | Precision | Recall | F1 | Judge-only precision | Judge-only recall "
            "| Pairs judged | Tokens on sample | Seconds on sample |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for r in runs:
            a = runs[r].get("article")
            if not a:
                continue
            e, j, c = a["end_to_end"], a["judge_only"], a.get("cost_on_sample") or {}
            tokens = (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
            out.append(
                f"| {name[r]} | {_ci(e['precision'], e['precision_ci'])} "
                f"| {_ci(e['recall'], e['recall_ci'])} | {_ci(e['f1'], e['f1_ci'])} "
                f"| {_ci(j['precision'], j['precision_ci'])} "
                f"| {_ci(j['recall'], j['recall_ci'])} "
                f"| {a['pairs_judged']} | {tokens:,} | {c.get('seconds', 0):,.0f} |"
            )
        if status:
            widths = status.get("half_widths") or {}
            out.append(
                f"\nStopping rule: the baseline's 95% intervals must reach "
                f"\u00b1{status.get('target', 0.05):.2f}. "
                f"Now \u00b1{widths.get('precision') or 0:.2f} "
                f"on precision and \u00b1{widths.get('recall') or 0:.2f} on recall, over "
                f"{status.get('positives')} positive pairs: "
                f"{'met' if status.get('target_met') else 'not met, keep labelling'}."
            )

    comparisons = report.get("comparisons") or {}
    if comparisons:
        out.append(
            "\n## Against the baseline\n\nAn arm counts as an improvement only if the "
            "paired interval on its F1 difference lies above zero.\n\n"
            "| Arm | F1 difference | Improvement | Discordant pairs | Only baseline right "
            "| Only arm right | McNemar p |\n| --- | --- | --- | --- | --- | --- | --- |"
        )
        for r, comparison in comparisons.items():
            f1, mc = comparison["f1"], comparison["mcnemar"]
            p = "-" if mc.get("p_value") is None else f"{mc['p_value']:.3f}"
            out.append(
                f"| {r} | {_ci(f1['difference'], f1['difference_ci'])} "
                f"| {'yes' if f1['improvement'] else 'no'} | {mc['discordant']} "
                f"| {mc['only_a_correct']} | {mc['only_b_correct']} | {p} |"
            )
        for r, comparison in comparisons.items():
            h = comparison.get("hierarchy")
            if not h:
                continue
            out.append(
                f"\n### Hierarchy diagnostics, run {r}\n\n"
                "Recall at each level counts only classes whose parent was answered yes, "
                "so a miss shows at the level where it happened.\n\n"
                "| Level | Recall | Hits / positives |\n| --- | --- | --- |"
            )
            for level, row in h["recall_by_level"].items():
                value = "-" if row.get("recall") is None else f"{row['recall']:.2f}"
                out.append(f"| {level} | {value} | {row['hits']}/{row['n']} |")
            cov = h["coverage"]
            out.append(
                f"\nParent classes answered yes: {h['internal_answered_yes']}, of which "
                f"{h['internal_false_positives']} with no labelled leaf below "
                "(gap candidates). "
                f"Correct leaf matches: {cov['true_positives']}, of which "
                f"{cov['beyond_baseline_retrieval']} on pairs the baseline's retrieval had "
                "not selected (coverage, not structure)."
            )

    out.append(
        "\n## Calendar\n\nVerified counts a detection only once a person has confirmed it "
        "is the calendar's event; until matches are reviewed, verified rates stay at "
        "zero, so the raw rate is shown beside each. Rates carry Wilson 95% "
        "intervals.\n\n"
        "| Run | | Events found | Precursors found | False alarms on controls "
        "| Verdicts | Tokens | Seconds |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in runs:
        cal, cost = runs[r]["calendar"], runs[r]["cost"]
        v = cal.get("verified") or {}
        tokens = cost["input_tokens"] + cost["output_tokens"]
        out.append(
            f"| {name[r]} | raw | {_rate_cell(cal.get('event_recall'))} "
            f"| {_rate_cell(cal.get('precursor_recall'))} "
            f"| {_rate_cell(cal.get('false_alarm_rate'))} "
            f"| {cost['pairs']:,} | {tokens:,} | {cost['seconds']:,.0f} |"
        )
        out.append(
            f"| | verified | {_rate_cell(v.get('event_recall'))} "
            f"| {_rate_cell(v.get('precursor_recall'))} "
            f"| {_rate_cell(v.get('false_alarm_rate'))} | | | |"
        )
    return "\n".join(out) + "\n"
