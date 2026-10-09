"""The judge loop.

Asks a model, for each selected `(document, concept)` pair, whether the document
evidences the concept. Two properties matter more than the asking:

**Resume.** A verdict is written per pair as it completes, and a restart skips
pairs already recorded. A run that dies two thirds of the way through a corpus
must not re-spend the first two thirds — with a local model that is wall-clock
time, with a hosted one it is money.

**Failures are recorded, not raised.** One unparseable response must not end a
run. The pair gets a verdict row with an error and the loop continues, which also
makes the failure countable afterwards instead of invisible. A liveness check
that every pair produced a parseable verdict is a separate, cheap gate — a
malformed-output bug is invisible to accuracy metrics until you look for it.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Locus, Run, Verdict
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.batched import judge_batched
from hontology.pipeline.judge.context import JudgeStats, Judging
from hontology.pipeline.judge.extract import judge_extract
from hontology.pipeline.judge.hierarchical import judge_hierarchical
from hontology.pipeline.judge.pairs import judge_pairs
from hontology.pipeline.judge.parse import AGGREGATIONS, aggregate, parse_batch, parse_verdict
from hontology.pipeline.judge.providers.base import GenerationConfig, ProviderError
from hontology.pipeline.judge.providers.llamacpp import LlamaCppChatProvider
from hontology.pipeline.judge.providers.ollama import OllamaChatProvider
from hontology.pipeline.judge.providers.openrouter import OpenRouterChatProvider
from hontology.pipeline.retrieve import embed
from hontology.pipeline.runs import fingerprints

# Readers live in judge.parse; these names are kept here for existing callers.
__all__ = [
    "AGGREGATIONS",
    "JudgeStats",
    "PromptChanged",
    "aggregate",
    "get_provider",
    "judge_run",
    "liveness",
    "parse_batch",
    "parse_verdict",
]


def get_provider(name: str, routing: dict | None = None):
    settings = get_settings()
    if name == "ollama":
        return OllamaChatProvider(settings.ollama_host)
    if name == "llamacpp":
        return LlamaCppChatProvider(settings.llamacpp_host, settings.llamacpp_api_key)
    if name == "openrouter":
        return OpenRouterChatProvider(
            settings.openrouter_api_key, routing, base_url=settings.openrouter_base_url
        )
    raise ProviderError(f"unknown judge provider {name!r}")


class PromptChanged(RuntimeError):
    """A prompt id no longer renders the wording it stands for."""


def check_wording(run: Run, prompt_id: str) -> None:
    """Refuse to judge with wording other than what the id, and the run, stand for.

    A pinned prompt id must render exactly as pinned in prompts.lock.json, so an
    edit in place cannot pass as the old prompt. A run created under one wording
    is never continued under another, so one run's verdicts share one wording.
    """
    current = fingerprints.prompt_fingerprint(prompt_id)
    pinned = prompts.PINS.get(prompt_id)
    if pinned is not None and pinned != current:
        raise PromptChanged(
            f"prompt {prompt_id!r} renders as {current} but is pinned to {pinned}: its "
            "wording was edited in place. Restore it, or register the new wording "
            "under a new prompt id and pin that."
        )
    recorded = ((run.manifest or {}).get("versions") or {}).get("prompt")
    if recorded and recorded != current:
        raise PromptChanged(
            f"run {run.id} was created with prompt {prompt_id!r} rendering as "
            f"{recorded}; it now renders as {current}. Continuing would mix two "
            "wordings in one run: start a new run instead."
        )


def _already_judged(session: Session, run_id: int) -> set[tuple[int, int]]:
    return {
        (document_id, concept_id)
        for document_id, concept_id in session.execute(
            select(Verdict.document_id, Verdict.concept_id).where(Verdict.run_id == run_id)
        )
    }


def judge_run(
    session: Session,
    run_id: int,
    *,
    config: dict,
    judge_body_limit: int,
    limit: int | None = None,
    progress: object | None = None,
    document_ids: set[int] | None = None,
    by_score: bool = False,
) -> dict:
    """Judge every selected candidate for a run, skipping those already done.

    *document_ids* narrows the work to some documents, so a run can be judged a
    window at a time. *by_score* takes the highest retrieval scores first, which
    is what makes a *limit* a budget spent on the strongest candidates rather
    than on whichever documents happen to sort first.
    """
    judge_config = config["judge"]
    template = prompts.get(judge_config["prompt_id"])
    run = session.get(Run, run_id)
    assert run is not None
    check_wording(run, template.prompt_id)

    query = select(Candidate).where(Candidate.run_id == run_id, Candidate.selected.is_(True))
    if document_ids is not None:
        query = query.where(Candidate.document_id.in_(document_ids))
    order = (
        (Candidate.score.desc(), Candidate.document_id, Candidate.concept_id)
        if by_score
        else (Candidate.document_id, Candidate.concept_id)
    )
    candidates = list(session.scalars(query.order_by(*order)))

    judging = Judging(
        session=session,
        run_id=run_id,
        template=template,
        provider=get_provider(judge_config["provider"], routing=judge_config.get("routing")),
        judge_config=judge_config,
        generation=GenerationConfig.from_config(judge_config["generation"]),
        body_limit=judge_body_limit,
        iso2_to_locus={
            locus.iso2: locus.id
            for locus in session.scalars(select(Locus).where(Locus.iso2.is_not(None)))
            if locus.iso2
        },
        stats=JudgeStats(total=len(candidates)),
    )

    if template.mode in (prompts.EXTRACT, prompts.HIERARCHICAL):
        if template.mode == prompts.HIERARCHICAL:
            return judge_hierarchical(
                judging,
                ontology_id=run.ontology_id,
                candidates=candidates,
                limit=limit,
                progress=progress,
            )
        embedder = None
        if template.leaf_top_k is not None:
            embed_provider = config.get("candidates", {}).get("embed_provider", "ollama")
            embedder = embed.get_provider(embed_provider)
        return judge_extract(
            judging,
            ontology_id=run.ontology_id,
            candidates=candidates,
            limit=limit,
            progress=progress,
            embedder=embedder,
        )

    done = _already_judged(session, run_id)
    if template.mode == prompts.PER_DOCUMENT:
        return judge_batched(
            judging, candidates=candidates, done=done, limit=limit, progress=progress
        )
    return judge_pairs(
        judging, candidates=candidates, done=done, limit=limit, progress=progress
    )


def liveness(session: Session, run_id: int) -> dict:
    """Did every pair produce a usable verdict?

    Checked separately from accuracy because a malformed-output bug does not move
    precision or recall — it removes pairs from the denominator entirely, which
    looks like nothing at all until you count.
    """
    verdicts = list(session.scalars(select(Verdict).where(Verdict.run_id == run_id)))
    errored = [v for v in verdicts if v.error]
    unparsed = [v for v in verdicts if v.matched is None and not v.error]

    return {
        "verdicts": len(verdicts),
        "errors": len(errored),
        "unparsed": len(unparsed),
        "clean": len(verdicts) - len(errored) - len(unparsed),
        "ok": not errored and not unparsed,
        "sample_errors": [str(v.error)[:120] for v in errored[:5]],
    }
