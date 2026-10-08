"""The scored runs of an ontology, end to end, computed live.

Every run is scored on the same labelled sample, the way the arms report scores
it (`arms.run_on_sample`): weighted end-to-end precision, recall and F1 with
article-bootstrap intervals, plus what it cost on those articles. Nothing is
cached, so a new label or run shows at once.

A run is only comparable on the sample if it processed the sample's articles,
so each row says how many of the labelled articles the run's retrieval covered.
A run made for another calendar covers few, and its recall there says nothing
about the run.

Calendar recall is per run and slow (every window is walked), so it is separate.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Candidate, Run, Verdict
from hontology.evalkit import arms, calendar, versions


def coverage(session: Session, run: Run, document_ids: set[int]) -> int:
    """How many of *document_ids* the run itself worked on.

    A run that judged a sample covers the prefix of its own sample it judged; one
    that reused another run's retrieval covers what it judged; one that did its
    own retrieval covers what that retrieval processed (judging less of it, under
    a budget, is the run's own doing and counts against it).
    """
    if not document_ids:
        return 0
    sample = (run.manifest or {}).get("sample")
    if sample:
        try:
            order = json.loads(Path(sample["manifest"]).read_text(encoding="utf-8"))["order"]
            judged = {entry["document_id"] for entry in order[: sample.get("judged_first")]}
            return len(judged & document_ids)
        except (OSError, ValueError, KeyError):
            pass  # the manifest moved: fall back to what was judged
    wanted = sorted(document_ids)
    source = versions.source_run(session, run)
    if source.id != run.id or sample:
        rows = select(Verdict.document_id).where(
            Verdict.run_id == run.id, Verdict.document_id.in_(wanted)
        )
    else:
        rows = select(Candidate.document_id).where(
            Candidate.run_id == run.id, Candidate.document_id.in_(wanted)
        )
    return len(set(session.scalars(rows.distinct())))


def stage_versions(session: Session, run: Run) -> dict:
    retrieval = versions.retrieval_version(session, run)
    judge = versions.judge_version(session, run)
    return {
        "filter": versions.filter_versions(run),
        "retrieval": retrieval["version"],
        "retrieval_source_run": retrieval["source_run"],
        "judge": judge["version"],
        "prompt_id": judge["prompt_id"],
        "model": judge["model"],
    }


def rows(
    session: Session,
    ontology_id: int,
    manifest: dict,
    labels: dict | None,
    *,
    include_partial: bool = False,
) -> dict:
    """Every judged run, scored on the sample if it worked on the whole of it.

    A run that covered only part of the sample is listed with its coverage but
    not scored unless *include_partial*: its scores there say nothing about it,
    and the bootstrap behind each score is what makes this slow.
    """
    sample = arms.labelled_sample(session, ontology_id, manifest, labels)
    documents = {doc for doc, _ in sample["truth"]}
    judged = set(
        session.scalars(
            select(Verdict.run_id).distinct().join(Run).where(Run.ontology_id == ontology_id)
        )
    )
    out = []
    for run in session.scalars(select(Run).where(Run.id.in_(judged)).order_by(Run.id.desc())):
        covered = coverage(session, run, documents)
        article: dict = {}
        if include_partial or covered == len(documents):
            # Judge-only scores are left to the run's own view (`run_on_sample` in
            # full): the table does not show them, and they cost as much again.
            scored = arms.run_on_sample(
                session, run.id, manifest, labels=labels, judge_only=False
            )
            article = scored.get("article") or {}
        out.append(
            {
                "run_id": run.id,
                "name": run.name,
                "status": run.status,
                "versions": stage_versions(session, run),
                "covered": covered,
                "end_to_end": article.get("end_to_end"),
                "judge_only": article.get("judge_only"),
                "pairs_judged": article.get("pairs_judged"),
                "cost": article.get("cost_on_sample"),
                "calendar": ((run.manifest or {}).get("documents") or {}).get("calendar"),
            }
        )
    return {
        "labelled_articles": len(documents),
        "labelled_prefix": sample["prefix"],
        "out_of_turn": len(sample["out_of_turn"]),
        "rows": out,
    }


def calendar_recall(session: Session, run_id: int, calendar_path: str | None = None) -> dict:
    """A run's calendar scores, on its own calendar unless another is given."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"run {run_id} does not exist")
    path = calendar_path or ((run.manifest or {}).get("documents") or {}).get("calendar")
    if not path:
        raise ValueError(f"run {run_id} did not work through a calendar; name one")
    result = calendar.evaluate(session, run_id, calendar.load(Path(path)))
    summary = result["summary"]
    return {
        "calendar": path,
        "event_recall": summary.get("event_recall"),
        "precursor_recall": summary.get("precursor_recall"),
        "false_alarm_rate": summary.get("false_alarm_rate"),
        "positives_lost_at": summary.get("positives_lost_at"),
        "verified": summary.get("verified"),
        "not_processed": len(result.get("not_processed") or []),
    }
