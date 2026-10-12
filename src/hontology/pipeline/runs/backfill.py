"""Backfilling the shown record for runs made before it existed.

A run judged before retrievals were stored has verdicts but no
`ArticleRetrieval` or `ShownLeaf` rows, so every metric reads it as having
shown its judge nothing. Its shown record is rebuilt from its verdicts:

- **One call, one transaction.** Every judge loop commits once per deciding
  call (an extraction run once per article), and a verdict's ``created_at`` is
  its transaction's start, so the rows of one call share it. Grouping a run's
  verdicts by (article, ``created_at``) recovers its calls.
- **Flat prompts:** each call's leaves are one offer.
- **Hierarchical:** each call over leaves is one offer; each call over parent
  classes is a routing call, kept as the retrieval's routing answers.
- **Extraction:** the leaves an article's events were weighed against are one
  offer for the article; which event each leaf was offered for was not stored.
  An article with no events wrote placeholder rows on the top-level classes to
  carry the extraction's cost, marked "no events extracted"; nothing was shown
  there, so it gets no offer. The marker is read rather than the stored events,
  which the earliest extraction runs did not keep.
- **Rows written without a call** (no article text, a failed extraction) were
  never shown, in any mode.

Backfilled runs get no retrieval key, so no new run adopts a rebuilt
retrieval: an extraction run's is per article, where adoption needs one offer
per event. The trace says how each retrieval was rebuilt.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import ArticleRetrieval, ExtractedEvent, Run, ShownLeaf, Verdict
from hontology.ontology import hierarchy
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.context import NO_BODY
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs import versions

EXTRACTION_FAILED = "event extraction failed"
# The evidence an extraction run wrote on its placeholder rows.
NO_EVENTS = "no events extracted"


@dataclass
class Backfill:
    """What backfilling one run wrote, or would write."""

    run_id: int
    strategy: str
    articles: int = 0
    offers: int = 0
    shown: int = 0
    # Leaf rows left out because no deciding call showed them.
    not_shown: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str, n: int = 1) -> None:
        self.not_shown[reason] = self.not_shown.get(reason, 0) + n


def unrecorded_runs(session: Session) -> list[Run]:
    """Runs with verdicts and no shown record, oldest first."""
    recorded = select(ArticleRetrieval.run_id).distinct()
    judged = select(Verdict.run_id).distinct()
    return list(
        session.scalars(
            select(Run).where(Run.id.in_(judged), Run.id.not_in(recorded)).order_by(Run.id)
        )
    )


def backfill_run(session: Session, run: Run, *, write: bool = True) -> Backfill:
    """Rebuild *run*'s shown record from its verdicts; adds rows unless *write* is
    False. The caller commits."""
    template = prompts.get(run_config.normalize(run.config or {})["judge"]["prompt_id"])
    mode = template.mode
    strategy = {
        prompts.HIERARCHICAL: "routing",
        prompts.EXTRACT: "extraction, ranked" if template.leaf_top_k else "extraction",
    }.get(mode, "embedding")
    result = Backfill(run.id, strategy)
    # The leaves at the run's ontology version; the current ones if it has no snapshot.
    leaves = {leaf["id"] for leaf in versions.leaf_wording(session, run)} or hierarchy.leaves(
        session, run.ontology_id
    )
    events = _events(session, run.id) if mode == prompts.EXTRACT else {}

    calls: dict[int, dict] = defaultdict(lambda: defaultdict(list))
    for verdict in session.scalars(
        select(Verdict).where(Verdict.run_id == run.id).order_by(Verdict.id)
    ):
        calls[verdict.document_id][verdict.created_at].append(verdict)

    for document_id, by_time in sorted(calls.items()):
        offers: list[list[int]] = []
        asked: list[dict] = []
        for _, rows in sorted(by_time.items()):
            shown = [v.concept_id for v in rows if v.concept_id in leaves]
            parents = [v for v in rows if v.concept_id not in leaves]
            if any(_never_called(v) for v in rows):
                result.skip("no call was made", len(shown))
                continue
            if mode == prompts.EXTRACT and any(v.evidence == NO_EVENTS for v in rows):
                result.skip("placeholder for an article with no events", len(shown))
                continue
            if mode == prompts.PER_PAIR:
                offers.extend([c] for c in shown)
            elif shown:
                offers.append(sorted(shown))
            if mode == prompts.HIERARCHICAL and parents:
                asked.append(_asked(parents))
        result.articles += 1
        result.offers += len(offers)
        result.shown += sum(len(o) for o in offers)
        if write:
            _write(
                session, run.id, document_id, strategy, offers, asked, events.get(document_id)
            )
    return result


def _never_called(verdict: Verdict) -> bool:
    error = verdict.error or ""
    return error == NO_BODY or error.startswith(EXTRACTION_FAILED)


def _asked(parents: list[Verdict]) -> dict:
    """One routing call over parent classes, as a stored retrieval keeps it."""
    return {
        "concept_ids": [v.concept_id for v in parents],
        "replies": None
        if any(v.error for v in parents)
        else {str(v.concept_id): {"matched": bool(v.matched)} for v in parents},
        "error": next((v.error for v in parents if v.error), None),
        "completion": None,
    }


def _events(session: Session, run_id: int) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for event in session.scalars(
        select(ExtractedEvent)
        .where(ExtractedEvent.run_id == run_id)
        .order_by(ExtractedEvent.document_id, ExtractedEvent.ordinal)
    ):
        out[event.document_id].append(
            {
                "found": {
                    "description": event.description,
                    "evidence": event.evidence or "",
                    "status": event.status or "",
                    "country": event.country or "",
                },
                "routed": list(event.routed_through),
                # Not stored per event before the shown record existed.
                "reached": [],
                "error": event.error,
            }
        )
    return out


def _write(
    session: Session,
    run_id: int,
    document_id: int,
    strategy: str,
    offers: list[list[int]],
    asked: list[dict],
    events: list[dict] | None,
) -> None:
    session.add(
        ArticleRetrieval(
            run_id=run_id,
            document_id=document_id,
            strategy=strategy,
            trace=[
                {
                    "step": "backfill",
                    "from": "verdicts, one call per transaction",
                    "offers": "per article" if events is not None else "per call",
                }
            ],
            asked=asked,
            events=events or [],
        )
    )
    session.add_all(
        ShownLeaf(run_id=run_id, document_id=document_id, concept_id=c, offer=n)
        for n, offer in enumerate(offers)
        for c in offer
    )
