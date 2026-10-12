"""Scoring a run against the ground-truth bank, one stage at a time.

The headline number nobody should report on its own is a single F1. A pipeline
misses an event for two entirely different reasons — retrieval never showed the
judge the concept, or the judge was shown it and said no — and those need
different fixes. So retrieval and judgment are scored separately, over the same
labels and one definition of what was shown (`evaluation.shown`), and the report
shows both. The embedding ranking is scored too, over its pre-cutoff pool, as a
diagnostic of retrieval's first step.

Every number here is computed over *trusted, non-stale* labels by default (see
`evaluation.labels.bank`), and the report says how many labels that was. A metric without
its denominator is not a measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.lookups import get_run
from hontology.db.models import Candidate, Observation, Verdict
from hontology.evaluation import shown as shown_module
from hontology.evaluation.article.verdicts import answered
from hontology.evaluation.labels import staleness
from hontology.evaluation.metrics.calibration import calibration_bins
from hontology.evaluation.metrics.confusion import Confusion
from hontology.evaluation.metrics.intervals import format_ci, format_num, with_intervals
from hontology.evaluation.metrics.retrieval import retrieval_metrics


@dataclass
class RunEvaluation:
    run_id: int
    run_name: str
    ontology_version: str
    n_labels: int
    judge: dict
    # What the judge was shown, against the labels.
    retrieval: dict
    # The embedding ranking, over its pool before the cutoff.
    embedding: dict
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
        for label in staleness.trusted_labels(
            session,
            ontology_id,
            include_machine=include_machine,
            include_stale=include_stale,
        )
    }


def verdict_map(session: Session, run_id: int) -> dict[tuple[int, int], bool]:
    """Only clean verdicts. An errored pair is not a prediction."""
    return answered(session, run_id)


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

    # --- judge: on the labelled pairs it was shown ------------------------
    shown = shown_module.shown_pairs(session, run_id) & set(truth)
    confusion = Confusion()
    for key, predicted in shown_module.judge_answers(session, run_id, shown).items():
        confusion.add(expected=truth[key], predicted=predicted)
    # Is the stated confidence a probability? Only pairs actually answered.
    calibration_pairs: list[tuple[float, bool]] = []
    for verdict in session.scalars(
        select(Verdict).where(
            Verdict.run_id == run_id,
            Verdict.error.is_(None),
            Verdict.matched.is_not(None),
            Verdict.confidence.is_not(None),
        )
    ):
        key = (verdict.document_id, verdict.concept_id)
        if key in shown and verdict.confidence is not None:
            calibration_pairs.append((verdict.confidence, bool(verdict.matched) == truth[key]))

    # --- embedding ranking ------------------------------------------------
    pool: dict[tuple[int, int], int] = {}
    selected: set[tuple[int, int]] = set()
    for candidate in session.scalars(select(Candidate).where(Candidate.run_id == run_id)):
        key = (candidate.document_id, candidate.concept_id)
        pool[key] = candidate.rank or 1
        if candidate.selected:
            selected.add(key)

    embedding = retrieval_metrics(truth, pool, selected)

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
        judge=with_intervals(confusion, seed=seed),
        retrieval=shown_module.retrieval_scores(session, run_id, truth),
        embedding=embedding.as_dict(),
        calibration=calibration_bins(calibration_pairs),
        liveness=_liveness(session, run_id),
        observations={
            "total": len(observations),
            "detected": detected,
            "detection_rate": (detected / len(observations)) if observations else None,
        },
    )


def _liveness(session: Session, run_id: int) -> dict:
    from hontology.pipeline.judge.run import liveness

    return liveness(session, run_id)


def format_report(evaluation: RunEvaluation) -> str:
    """A plain-text report, with denominators next to every rate."""
    judge = evaluation.judge
    retrieval = evaluation.retrieval
    embedding = evaluation.embedding
    lines: list[str] = []

    lines.append(f"run {evaluation.run_id}  {evaluation.run_name}")
    lines.append(f"ontology version   {evaluation.ontology_version}")
    lines.append(f"labels used        {evaluation.n_labels}")
    lines.append("")

    if not judge["n"]:
        lines.append("JUDGE: no labelled pair was shown to this run's judge.")
    else:
        lines.append(f"JUDGE  (n={judge['n']} labelled pairs shown; a failed call is a no)")
        lines.append(
            f"  tp={judge['tp']}  fp={judge['fp']}  tn={judge['tn']}  fn={judge['fn']}"
        )
        for name, ci_key in (
            ("precision", "precision_ci"),
            ("recall", "recall_ci"),
            ("f1", "f1_ci"),
        ):
            value = format_num(judge[name])
            lines.append(f"  {name:<10} {value}  {format_ci(*judge[ci_key])}")
    lines.append("")

    lines.extend(_retrieval_lines(retrieval))
    lines.append("")

    lines.append("EMBEDDING RANKER  (the pool before the cutoff: retrieval's first step)")
    lines.append(
        f"  candidates {embedding['total_candidates']}, "
        f"labelled {embedding['labelled_candidates']}"
    )
    coverage = embedding["coverage"]
    precision = embedding["precision"]
    lines.append(f"  precision  {format_num(precision)}")
    lines.append(
        f"  coverage   {coverage:.3f}   <- precision's denominator; unlabelled "
        f"candidates are unknown, not wrong"
        if coverage is not None
        else "  coverage   —"
    )
    for k, value in sorted(embedding["recall_at_k"].items()):
        lines.append(f"  recall@{k:<3} {format_num(value)}")
    lines.append(f"  mrr        {format_num(embedding['mrr'])}")
    cutoff = embedding["cutoff_recall"]
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
        lines.append("  ! a failed call counts as a no on the pairs it was shown, so it")
        lines.append("    lowers the judge's recall: check this before the metrics")

    if evaluation.observations["total"]:
        rate = evaluation.observations["detection_rate"]
        lines.append("")
        lines.append(
            f"OBSERVATIONS  {evaluation.observations['detected']}/"
            f"{evaluation.observations['total']} detected"
            + (f"  ({rate:.3f})" if rate is not None else "")
        )
    return "\n".join(lines)


def _retrieval_lines(retrieval: dict) -> list[str]:
    """What the judge was shown, against the labels."""
    if not retrieval["shown_record"]:
        return [
            "RETRIEVAL: no record of what this run showed its judge (made before it existed)."
        ]
    lines = [
        f"RETRIEVAL  (what the judge was shown: {retrieval['shown_pairs']} pairs, "
        f"{retrieval['labelled_shown']} labelled)"
    ]
    lines.append(
        f"  recall     {format_num(retrieval['recall'])}  {format_ci(*retrieval['recall_ci'])}"
        f"   <- {retrieval['positives_shown']} of {retrieval['positives']} labelled "
        "positives on the articles it ranked"
    )
    lines.append(
        f"  precision  {format_num(retrieval['precision'])}  "
        f"{format_ci(*retrieval['precision_ci'])}"
    )
    coverage = retrieval["coverage"]
    lines.append(
        f"  coverage   {coverage:.3f}   <- precision's denominator; unlabelled pairs are "
        "unknown, not wrong"
        if coverage is not None
        else "  coverage   —"
    )
    return lines
