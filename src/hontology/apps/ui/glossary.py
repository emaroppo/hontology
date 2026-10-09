"""One place that explains every metric.

Streamlit renders a `(?)` tooltip for the `help=` argument on `st.metric` and
friends. Keeping the wording here means a metric is explained the same way on
every page — and, more usefully, that the explanation says what the number can
*mislead* you about, not just what it is.

A dashboard whose numbers are unexplained gets read optimistically. These
definitions are deliberately blunt about denominators and small samples, because
that is where a plausible number stops meaning anything.
"""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # --- judge ---------------------------------------------------------------
    "precision": (
        "Of the pairs the judge called a match, the share that really were. "
        "Computed only over pairs carrying a **trusted** label — machine "
        "proposals nobody has reviewed are excluded by default."
    ),
    "recall": (
        "Of the pairs that really were a match, the share the judge found. "
        "Reads as `—` rather than 0 when the label set contains no positives: "
        "undefined is not the same as zero."
    ),
    "f1": (
        "Harmonic mean of precision and recall. Shown with a bootstrap interval "
        "because a point estimate over a few hundred labels invites over-reading "
        "— a ten-point gap on thirty pairs is noise."
    ),
    "confusion": (
        "True positives / false positives / true negatives / false negatives. "
        "The counts matter more than the rates when the sample is small, and "
        "they are what lets rates be recomputed correctly for any grouping."
    ),
    "n_labels": (
        "How many labelled pairs the metrics were computed over. A rate without "
        "this number is not a measurement."
    ),
    # --- retrieval -----------------------------------------------------------
    "retrieval_precision": (
        "Of the candidates retrieval proposed, the share that carry a positive "
        "label. Measured **only over labelled candidates** — see coverage."
    ),
    "coverage": (
        "The share of proposed candidates that carry any label at all, and so "
        "the denominator retrieval precision is computed over. An unlabelled "
        "candidate is *unknown*, never counted as wrong. Low coverage means the "
        "precision figure above it is based on very little."
    ),
    "recall_at_k": (
        "Of the known positive pairs, the share whose concept appeared in the "
        "top k of its document's pre-cutoff pool. Measures *ranking*, "
        "independently of where the cutoff happened to fall."
    ),
    "mrr": (
        "Mean reciprocal rank of the correct concept within the pre-cutoff pool. "
        "1.0 means it was always ranked first; 0.5 means second on average."
    ),
    "cutoff_recall": (
        "Of the positives retrieval *did* rank, the share that survived the "
        "cutoff. Separates 'never surfaced it' from 'surfaced it and then "
        "discarded it' — two failures with completely different fixes."
    ),
    # --- confidence and consistency -----------------------------------------
    "confidence": (
        "The model's self-reported probability that its verdict is correct. "
        "Treat with suspicion: check the calibration table before believing it. "
        "Some models emit a near-constant value regardless of what they decided, "
        "which carries no information at all."
    ),
    "vote_fraction": (
        "When a run takes several samples, the share that voted with the "
        "majority. A more honest confidence than the model's self-report: asked "
        "five times and answering yes three times is genuine uncertainty."
    ),
    "calibration": (
        "A calibrated model is right about 70% of the time when it says 0.7. "
        "Flat or empty bins mean its confidence carries no signal — the "
        "labelling queue detects this and ignores such a run's confidence."
    ),
    "determinism": (
        "Whether re-running an identical config produces identical verdicts. If "
        "not, run-to-run noise is a floor under every A/B difference, and a "
        "smaller gap than that floor cannot be attributed to a config change."
    ),
    "cross_document_agreement": (
        "Whether verdicts agree across several documents covering the same "
        "concept in the same place. Disagreement measures instability that no "
        "accuracy number shows."
    ),
    # --- comparison ----------------------------------------------------------
    "mcnemar": (
        "A paired test over the pairs both runs judged. Only the pairs they "
        "answered *differently* carry information about which is better; pairs "
        "both got right, or both got wrong, say nothing."
    ),
    "discordant": (
        "Pairs where the two runs disagree. Few discordant pairs means a "
        "difference cannot be significant however large the gap in rates looks."
    ),
    "p_value": (
        "Probability of seeing a split this lopsided if the two runs were "
        "equally good. Small means the difference is real; it says nothing "
        "about whether it is *large*."
    ),
    # --- labels --------------------------------------------------------------
    "trusted_labels": (
        "Labels counted as ground truth: human, adjudicated, or imported. "
        "Machine proposals are excluded until a human confirms or flips them, "
        "because scoring a model against another model's unreviewed labels "
        "measures agreement, not correctness."
    ),
    "pending_adjudication": (
        "Machine proposals awaiting review. They do not count toward any metric "
        "until confirmed or flipped."
    ),
    "stale_labels": (
        "Labels whose concept was reworded after they were made, so they answer "
        "a question no longer being asked. Excluded from metrics by default; "
        "re-adjudicating one against the current wording restores it."
    ),
    "observations": (
        "Known occurrences — 'this concept happened here on this date'. Positive "
        "only, and independent of any pipeline run, so they stay valid when the "
        "pipeline changes."
    ),
    # --- pipeline ------------------------------------------------------------
    "liveness": (
        "Whether every judged pair produced a usable verdict. Checked before any "
        "rate, because a malformed-output bug does not move precision or recall "
        "— it silently removes pairs from the denominator."
    ),
    "candidates_key": (
        "Content hash of the retrieval settings. Two runs sharing it share their "
        "retrieval artifact, so the second reuses it instead of re-embedding."
    ),
    "judge_key": (
        "Content hash of the judge settings, chained to the candidates key. "
        "Changing only the prompt leaves the candidates key untouched."
    ),
    "lag_slices": (
        "How many 15-minute feed slices behind the ingester is. `null` before "
        "the first ingest — an install that has never run is idle, not current."
    ),
    "filter_coverage": (
        "The share of unfetched documents the code filter would admit. Only "
        "meaningful for an ontology whose concepts are mapped onto the code "
        "system."
    ),
}


def help_for(key: str) -> str | None:
    """Tooltip text for a metric, or None if it has no entry."""
    return GLOSSARY.get(key)


def describe(key: str) -> str:
    """Tooltip text, falling back to the key so a gap is visible rather than blank."""
    return GLOSSARY.get(key, f"(no glossary entry for {key!r})")
