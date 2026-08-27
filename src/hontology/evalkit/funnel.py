"""Where the volume went.

Per-stage metrics answer "how good is this stage". The funnel answers a
different and often more urgent question: *why did this run produce so little?*
A pipeline that returns four detections from a thousand documents might be
precise, or the scraper might be failing, or the cutoff might be discarding
everything — and precision tells you nothing about which.

Each step reports its count, its share of the previous step, and its share of
the top. The step-over-step number is the one that localises a problem: a stage
passing 3% of what reached it is where to look, regardless of how small the
final total is.

**Attrition is not failure.** Most of it is the pipeline working — a filter is
*supposed* to discard, and most articles genuinely do not evidence most
concepts. The funnel is a diagnostic, not a scorecard, so each step carries a
note saying what a healthy drop looks like there.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Document, Run, Verdict


@dataclass
class Step:
    name: str
    count: int
    note: str = ""

    def as_dict(self, previous: int | None, top: int) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "of_previous": (self.count / previous) if previous else None,
            "of_top": (self.count / top) if top else None,
            "note": self.note,
        }


def funnel(session: Session, run_id: int) -> dict:
    """Document and pair counts at each stage of one run."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")

    corpus = session.scalar(select(func.count(Document.id))) or 0
    fetched = (
        session.scalar(select(func.count(Document.id)).where(Document.fetched_at.is_not(None)))
        or 0
    )
    usable = (
        session.scalar(
            select(func.count(Document.id)).where(
                Document.body_path.is_not(None), Document.is_junk.is_(False)
            )
        )
        or 0
    )

    considered = (
        session.scalar(
            select(func.count(func.distinct(Candidate.document_id))).where(
                Candidate.run_id == run_id
            )
        )
        or 0
    )
    pool = (
        session.scalar(select(func.count(Candidate.id)).where(Candidate.run_id == run_id)) or 0
    )
    selected = (
        session.scalar(
            select(func.count(Candidate.id)).where(
                Candidate.run_id == run_id, Candidate.selected.is_(True)
            )
        )
        or 0
    )

    judged = (
        session.scalar(
            select(func.count(Verdict.id)).where(
                Verdict.run_id == run_id, Verdict.error.is_(None), Verdict.matched.is_not(None)
            )
        )
        or 0
    )
    errored = (
        session.scalar(
            select(func.count(Verdict.id)).where(
                Verdict.run_id == run_id, Verdict.error.is_not(None)
            )
        )
        or 0
    )
    matched = (
        session.scalar(
            select(func.count(Verdict.id)).where(
                Verdict.run_id == run_id, Verdict.matched.is_(True), Verdict.error.is_(None)
            )
        )
        or 0
    )

    documents = [
        Step("documents ingested", corpus, "everything the feed has yielded"),
        Step(
            "fetched",
            fetched,
            "a large drop here means the scrape budget, not a failure",
        ),
        Step(
            "usable text",
            usable,
            "the rest were paywalls, dead pages or nav dumps caught by the quality gate",
        ),
        Step(
            "considered by this run",
            considered,
            "bounded by the run's document limit and any ingest filter",
        ),
    ]
    pairs = [
        Step("candidate pairs (pool)", pool, "every concept ranked per document"),
        Step(
            "selected by the cutoff",
            selected,
            "the cutoff is supposed to discard most of the pool",
        ),
        Step("judged", judged, "pairs the model returned a usable verdict for"),
        Step(
            "matched",
            matched,
            "most articles genuinely do not evidence most concepts; a low share is normal",
        ),
    ]

    def render(steps: list[Step]) -> list[dict]:
        top = steps[0].count
        out: list[dict] = []
        previous: int | None = None
        for step in steps:
            out.append(step.as_dict(previous, top))
            previous = step.count
        return out

    return {
        "run_id": run_id,
        "run_name": run.name,
        "documents": render(documents),
        "pairs": render(pairs),
        "errored_verdicts": errored,
        # A judged count short of selected, with no errors to explain it, means
        # the run stopped early rather than the model failing.
        "unjudged_selected": max(0, selected - judged - errored),
    }


def format_funnel(result: dict) -> str:
    lines = [f"run {result['run_id']}  {result['run_name']}", ""]

    for title, key in (("DOCUMENTS", "documents"), ("PAIRS", "pairs")):
        lines.append(title)
        for step in result[key]:
            of_previous = (
                f"{step['of_previous']:>6.1%}" if step["of_previous"] is not None else "      "
            )
            of_top = f"{step['of_top']:>6.1%}" if step["of_top"] is not None else "      "
            lines.append(
                f"  {step['name']:<26} {step['count']:>8}  {of_previous} of prev  "
                f"{of_top} of top"
            )
            if step["note"]:
                lines.append(f"  {'':<26} {'':>8}  {step['note']}")
        lines.append("")

    if result["errored_verdicts"]:
        lines.append(
            f"! {result['errored_verdicts']} verdict(s) errored — these leave the "
            "denominator without moving any rate"
        )
    if result["unjudged_selected"]:
        lines.append(
            f"! {result['unjudged_selected']} selected pair(s) were never judged — "
            "the run stopped early rather than the model failing"
        )
    return "\n".join(lines)
