"""Comparing two runs, and checking a run against itself.

**Comparison is paired.** Two runs judged the same pairs, so the right test uses
that: only the pairs where they disagree carry information about which is better,
and an unpaired comparison throws that away. McNemar's test over the discordant
pairs, alongside intervals on each run's own metrics.

**Determinism is the gate, not a nicety.** If re-running an identical config
produces different verdicts, then no A/B delta can be attributed to the config
change — the run-to-run noise is a floor under every difference you might want to
report. So it is checked first, and its answer bounds how any comparison should
be read.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from hontology.db.models import Run
from hontology.evalkit import evaluate as evaluate_module
from hontology.evalkit import metrics as metric_lib
from hontology.evalkit.article import clean_verdicts


@dataclass
class Comparison:
    run_a: int
    run_b: int
    name_a: str
    name_b: str
    n_shared_labelled: int
    metrics_a: dict
    metrics_b: dict
    paired: dict
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def compare_runs(
    session: Session,
    run_a_id: int,
    run_b_id: int,
    *,
    alpha: float = 0.05,
    include_machine: bool = False,
    include_stale: bool = False,
) -> Comparison:
    run_a = session.get(Run, run_a_id)
    run_b = session.get(Run, run_b_id)
    if run_a is None or run_b is None:
        raise LookupError("both runs must exist")
    if run_a.ontology_id != run_b.ontology_id:
        raise ValueError("runs belong to different ontologies and are not comparable")

    truth = evaluate_module.label_map(
        session,
        run_a.ontology_id,
        include_machine=include_machine,
        include_stale=include_stale,
    )
    verdicts_a = evaluate_module.verdict_map(session, run_a_id)
    verdicts_b = evaluate_module.verdict_map(session, run_b_id)

    shared = [k for k in truth if k in verdicts_a and k in verdicts_b]

    # Score both over the SAME shared subset, otherwise the comparison is
    # between different test sets wearing the same label.
    confusion_a = metric_lib.Confusion()
    confusion_b = metric_lib.Confusion()
    for key in shared:
        confusion_a.add(expected=truth[key], predicted=verdicts_a[key])
        confusion_b.add(expected=truth[key], predicted=verdicts_b[key])

    paired = metric_lib.mcnemar(truth, verdicts_a, verdicts_b)
    verdict = _interpret(paired, run_a.name, run_b.name, alpha=alpha)

    return Comparison(
        run_a=run_a_id,
        run_b=run_b_id,
        name_a=run_a.name,
        name_b=run_b.name,
        n_shared_labelled=len(shared),
        metrics_a=metric_lib.with_intervals(confusion_a),
        metrics_b=metric_lib.with_intervals(confusion_b),
        paired=paired.as_dict(),
        verdict=verdict,
    )


def _interpret(
    paired: metric_lib.McNemarResult, name_a: str, name_b: str, *, alpha: float
) -> str:
    """State plainly what the numbers do and do not support."""
    if paired.n_pairs == 0:
        return "no labelled pairs judged by both runs — nothing to compare yet"
    if paired.discordant_count == 0:
        return f"identical on all {paired.n_pairs} shared labelled pairs"

    p = paired.p_value
    if p is None:
        return "not enough disagreement to test"
    better, worse = (
        (name_a, name_b) if paired.only_a_correct > paired.only_b_correct else (name_b, name_a)
    )
    if p < alpha:
        return f"{better} beats {worse} (p={p:.4f}, {paired.discordant_count} discordant)"
    return (
        f"no significant difference (p={p:.4f}, only "
        f"{paired.discordant_count} discordant pairs) — the apparent gap is "
        f"within noise at this sample size"
    )


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def determinism(session: Session, run_a_id: int, run_b_id: int) -> dict:
    """Compare two executions of the same config.

    If this is not 1.0, run-to-run noise is a floor under every A/B delta, and a
    difference smaller than that floor is not attributable to anything.
    """
    a = evaluate_module.verdict_map(session, run_a_id)
    b = evaluate_module.verdict_map(session, run_b_id)
    shared = set(a) & set(b)
    flipped = sorted(key for key in shared if a[key] != b[key])

    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": len(shared),
        "agreement": (len(shared) - len(flipped)) / len(shared) if shared else None,
        "flipped": len(flipped),
        "reproducible": not flipped and bool(shared),
        "sample_flips": flipped[:10],
    }


def cross_document_agreement(session: Session, run_id: int) -> dict:
    """Do verdicts agree across documents covering the same concept and place?

    Several outlets reporting one event should produce one answer. Where they
    do not, the disagreement is a measure of instability that no accuracy number
    shows — and those pairs are the best labelling candidates.
    """
    groups: dict[tuple[int, int | None], list[bool]] = defaultdict(list)
    for verdict in clean_verdicts(session, run_id):
        groups[(verdict.concept_id, verdict.locus_id)].append(bool(verdict.matched))

    multi = 0
    split = 0
    agreement_sum = 0.0
    worst: list[dict] = []

    for (concept_id, locus_id), flags in groups.items():
        if len(flags) < 2:
            continue
        multi += 1
        positives = sum(flags)
        majority = max(positives, len(flags) - positives)
        agreement = majority / len(flags)
        agreement_sum += agreement
        if agreement < 1.0:
            split += 1
            worst.append(
                {
                    "concept_id": concept_id,
                    "locus_id": locus_id,
                    "documents": len(flags),
                    "matched": positives,
                    "agreement": round(agreement, 3),
                }
            )

    worst.sort(key=lambda row: (row["agreement"], -row["documents"]))
    return {
        "multi_document_groups": multi,
        "split_groups": split,
        "mean_agreement": round(agreement_sum / multi, 3) if multi else None,
        "worst": worst[:10],
    }
