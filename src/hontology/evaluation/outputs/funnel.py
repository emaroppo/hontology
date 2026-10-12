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

from hontology.db.lookups import count, get_run
from hontology.db.models import ArticleRetrieval, Candidate, Document, Verdict
from hontology.evaluation import shown as shown_module


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
    run = get_run(session, run_id)
    documents = _document_steps(session, run_id)
    shown = shown_module.shown_pairs(session, run_id)
    pairs = _pair_steps(session, run_id, shown)
    recorded = shown_module.has_record(session, run_id)
    return {
        "run_id": run_id,
        "run_name": run.name,
        "documents": _render(documents),
        "pairs": _render(pairs),
        "shown_record": recorded,
        # Shown pairs whose call failed: they count as a no from the judge.
        "errored_verdicts": pairs[2].count - pairs[3].count,
        # Articles the cutoff let through that were never retrieved for: the
        # run stopped early, or a budget stopped it, rather than the model failing.
        "unretrieved_documents": _unretrieved(session, run_id) if recorded else None,
    }


def _document_steps(session: Session, run_id: int) -> list[Step]:
    corpus = count(session, Document.id)
    fetched = count(session, Document.id, Document.fetched_at.is_not(None))
    usable = count(
        session, Document.id, Document.body_path.is_not(None), Document.is_junk.is_(False)
    )
    considered = count(
        session, func.distinct(Candidate.document_id), Candidate.run_id == run_id
    )
    return [
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


def _pair_steps(session: Session, run_id: int, shown: set) -> list[Step]:
    """Pool, selected, shown, judged and matched, in that order."""
    pool = count(session, Candidate.id, Candidate.run_id == run_id)
    selected = count(
        session, Candidate.id, Candidate.run_id == run_id, Candidate.selected.is_(True)
    )
    usable = _usable(session, run_id, shown)
    judged, matched = len(usable), sum(usable.values())
    return [
        Step("candidate pairs (pool)", pool, "every concept ranked per document"),
        Step(
            "selected by the cutoff",
            selected,
            "the cutoff is supposed to discard most of the pool",
        ),
        Step(
            "shown to the judge",
            len(shown),
            "what each deciding call was shown: retrieved, in every prompt mode",
        ),
        Step("judged", judged, "shown pairs the model returned a usable verdict for"),
        Step(
            "matched",
            matched,
            "most articles genuinely do not evidence most concepts; a low share is normal",
        ),
    ]


def _render(steps: list[Step]) -> list[dict]:
    top = steps[0].count
    out: list[dict] = []
    previous: int | None = None
    for step in steps:
        out.append(step.as_dict(previous, top))
        previous = step.count
    return out


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

    if not result["shown_record"]:
        lines.append(
            "! no record of what this run showed its judge: it was made before the "
            "record existed, so the shown, judged and matched steps are empty"
        )
    if result["errored_verdicts"]:
        lines.append(
            f"! {result['errored_verdicts']} shown pair(s) got no usable verdict — "
            "they count as a no, and liveness says why"
        )
    if result["unretrieved_documents"]:
        lines.append(
            f"! {result['unretrieved_documents']} article(s) the cutoff let through were "
            "never retrieved for — the run stopped early rather than the model failing"
        )
    return "\n".join(lines)


def _usable(session: Session, run_id: int, shown: set) -> dict[tuple[int, int], bool]:
    """Each shown pair with a usable verdict, and whether it matched."""
    rows = session.execute(
        select(Verdict.document_id, Verdict.concept_id, Verdict.matched).where(
            Verdict.run_id == run_id,
            Verdict.error.is_(None),
            Verdict.matched.is_not(None),
        )
    )
    return {(d, c): bool(m) for d, c, m in rows if (d, c) in shown}


def _unretrieved(session: Session, run_id: int) -> int:
    """Articles with a selected candidate but no stored retrieval."""
    retrieved = select(ArticleRetrieval.document_id).where(ArticleRetrieval.run_id == run_id)
    return count(
        session,
        func.distinct(Candidate.document_id),
        Candidate.run_id == run_id,
        Candidate.selected.is_(True),
        Candidate.document_id.not_in(retrieved),
    )
