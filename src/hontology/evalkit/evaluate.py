"""Scoring a run against the ground-truth bank, one stage at a time.

The headline number nobody should report on its own is a single F1. A pipeline
misses an event for two entirely different reasons — retrieval never surfaced the
concept, or the judge saw it and said no — and those need different fixes. So
retrieval and judgment are scored separately, over the same labels, and the
report shows both.

Every number here is computed over *trusted, non-stale* labels by default (see
`evalkit.labels`), and the report says how many labels that was. A metric without
its denominator is not a measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import get_run
from hontology.db.models import Candidate, Observation, Verdict
from hontology.evalkit import article
from hontology.evalkit import labels as label_service
from hontology.evalkit import metrics as metric_lib


@dataclass
class RunEvaluation:
    run_id: int
    run_name: str
    ontology_version: str
    n_labels: int
    judge: dict
    retrieval: dict
    calibration: list[dict]
    liveness: dict
    observations: dict

    def as_dict(self) -> dict:
        return asdict(self)


def label_map(
    session: Session,
    ontology_id: int,
    *,
    include_machine: bool = False,
    include_stale: bool = False,
) -> dict[tuple[int, int], bool]:
    return {
        (label.document_id, label.concept_id): label.matched
        for label in label_service.trusted_labels(
            session,
            ontology_id,
            include_machine=include_machine,
            include_stale=include_stale,
        )
    }


def verdict_map(session: Session, run_id: int) -> dict[tuple[int, int], bool]:
    """Only clean verdicts. An errored pair is not a prediction."""
    return article.answered(session, run_id)


def evaluate_run(
    session: Session,
    run_id: int,
    *,
    include_machine: bool = False,
    include_stale: bool = False,
    seed: int = 0,
) -> RunEvaluation:
    run = get_run(session, run_id)

    truth = label_map(
        session,
        run.ontology_id,
        include_machine=include_machine,
        include_stale=include_stale,
    )

    # --- judge ------------------------------------------------------------
    confusion = metric_lib.Confusion()
    calibration_pairs: list[tuple[float, bool]] = []
    for verdict in article.clean_verdicts(session, run_id):
        key = (verdict.document_id, verdict.concept_id)
        if key not in truth:
            continue
        expected = truth[key]
        confusion.add(expected=expected, predicted=bool(verdict.matched))
        if verdict.confidence is not None:
            calibration_pairs.append(
                (float(verdict.confidence), bool(verdict.matched) == expected)
            )

    # --- retrieval --------------------------------------------------------
    pool: dict[tuple[int, int], int] = {}
    selected: set[tuple[int, int]] = set()
    for candidate in session.scalars(select(Candidate).where(Candidate.run_id == run_id)):
        key = (candidate.document_id, candidate.concept_id)
        pool[key] = candidate.rank or 1
        if candidate.selected:
            selected.add(key)

    retrieval = metric_lib.retrieval_metrics(truth, pool, selected)

    # --- end to end -------------------------------------------------------
    observations = list(
        session.scalars(
            select(Observation)
            .join(Candidate, Candidate.concept_id == Observation.concept_id)
            .where(Candidate.run_id == run_id)
            .distinct()
        )
    )
    detected = 0
    for observation in observations:
        hit = session.scalar(
            select(Verdict).where(
                Verdict.run_id == run_id,
                Verdict.concept_id == observation.concept_id,
                Verdict.matched.is_(True),
                Verdict.locus_id == observation.locus_id,
            )
        )
        detected += 1 if hit is not None else 0

    return RunEvaluation(
        run_id=run.id,
        run_name=run.name,
        ontology_version=run.ontology_version,
        n_labels=len(truth),
        judge=metric_lib.with_intervals(confusion, seed=seed),
        retrieval=retrieval.as_dict(),
        calibration=metric_lib.calibration_bins(calibration_pairs),
        liveness=_liveness(session, run_id),
        observations={
            "total": len(observations),
            "detected": detected,
            "detection_rate": (detected / len(observations)) if observations else None,
        },
    )


def _liveness(session: Session, run_id: int) -> dict:
    from hontology.judge.run import liveness

    return liveness(session, run_id)


def format_report(evaluation: RunEvaluation) -> str:
    """A plain-text report, with denominators next to every rate."""
    judge = evaluation.judge
    retrieval = evaluation.retrieval
    lines: list[str] = []

    lines.append(f"run {evaluation.run_id}  {evaluation.run_name}")
    lines.append(f"ontology version   {evaluation.ontology_version}")
    lines.append(f"labels used        {evaluation.n_labels}")
    lines.append("")

    if not judge["n"]:
        lines.append("JUDGE: no labelled pairs were judged by this run.")
    else:
        lines.append(f"JUDGE  (n={judge['n']})")
        lines.append(
            f"  tp={judge['tp']}  fp={judge['fp']}  tn={judge['tn']}  fn={judge['fn']}"
        )
        for name, ci_key in (
            ("precision", "precision_ci"),
            ("recall", "recall_ci"),
            ("f1", "f1_ci"),
        ):
            shown = metric_lib.format_num(judge[name])
            lines.append(f"  {name:<10} {shown}  {metric_lib.format_ci(*judge[ci_key])}")
    lines.append("")

    lines.append("RETRIEVAL")
    lines.append(
        f"  candidates {retrieval['total_candidates']}, "
        f"labelled {retrieval['labelled_candidates']}"
    )
    coverage = retrieval["coverage"]
    precision = retrieval["precision"]
    lines.append(f"  precision  {metric_lib.format_num(precision)}")
    lines.append(
        f"  coverage   {coverage:.3f}   <- precision's denominator; unlabelled "
        f"candidates are unknown, not wrong"
        if coverage is not None
        else "  coverage   —"
    )
    for k, value in sorted(retrieval["recall_at_k"].items()):
        lines.append(f"  recall@{k:<3} {metric_lib.format_num(value)}")
    lines.append(f"  mrr        {metric_lib.format_num(retrieval['mrr'])}")
    cutoff = retrieval["cutoff_recall"]
    lines.append(
        f"  cutoff     {cutoff:.3f}   <- of positives that WERE ranked, how many survived"
        if cutoff is not None
        else "  cutoff     —"
    )
    lines.append("")

    live = evaluation.liveness
    lines.append(
        f"LIVENESS  {live['clean']}/{live['verdicts']} clean, "
        f"{live['errors']} error(s), {live['unparsed']} unparsed"
    )
    if not live["ok"]:
        lines.append("  ! a malformed-output bug shrinks the denominator without")
        lines.append("    moving precision or recall — check this before the metrics")

    if evaluation.observations["total"]:
        rate = evaluation.observations["detection_rate"]
        lines.append("")
        lines.append(
            f"OBSERVATIONS  {evaluation.observations['detected']}/"
            f"{evaluation.observations['total']} detected"
            + (f"  ({rate:.3f})" if rate is not None else "")
        )
    return "\n".join(lines)
