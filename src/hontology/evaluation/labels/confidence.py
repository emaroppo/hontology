"""Whether a run's confidence carries an uncertainty signal worth ranking on.

Many configurations emit degenerate confidence: always ≈0 or ≈1, or
near-constant. Such a run's numbers are ignored rather than read as uncertainty
(see `labels.queue`).
"""

from __future__ import annotations

import statistics

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Verdict

# Thresholds for deciding whether a run's confidence carries usable signal.
CONF_MIN_SAMPLES = 20
CONF_MIN_STD = 0.05
CONF_INTERIOR_LO = 0.15
CONF_INTERIOR_HI = 0.85
CONF_MIN_INTERIOR_FRAC = 0.05


def usable_confidence_runs(session: Session, run_ids: list[int]) -> set[int]:
    """Runs whose confidence distribution carries an uncertainty signal.

    Rejects too-few samples, near-constant output (a run that always says 0.5),
    and effectively binary output (a run that only ever says ~0 or ~1). Treating
    a degenerate distribution as uncertainty is worse than ignoring it: it
    actively misranks the queue.
    """
    usable: set[int] = set()
    for run_id in run_ids:
        values = [
            float(c)
            for (c,) in session.execute(
                select(Verdict.confidence).where(
                    Verdict.run_id == run_id,
                    Verdict.confidence.is_not(None),
                    Verdict.error.is_(None),
                )
            )
        ]
        if len(values) < CONF_MIN_SAMPLES:
            continue
        if statistics.pstdev(values) < CONF_MIN_STD:
            continue
        interior = sum(1 for v in values if CONF_INTERIOR_LO < v < CONF_INTERIOR_HI) / len(
            values
        )
        if interior < CONF_MIN_INTERIOR_FRAC:
            continue
        usable.add(run_id)
    return usable
