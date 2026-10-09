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

import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Concept, Document, Locus, Run, Verdict
from hontology.judge import prompts
from hontology.judge.batched import judge_batched, judge_hierarchical
from hontology.judge.context import NO_BODY, JudgeStats, Judging
from hontology.judge.extract import judge_extract
from hontology.judge.parse import AGGREGATIONS, aggregate, parse_batch, parse_verdict
from hontology.judge.providers.base import GenerationConfig, ProviderError
from hontology.judge.providers.llamacpp import LlamaCppChatProvider
from hontology.judge.providers.ollama import OllamaChatProvider
from hontology.judge.providers.openrouter import OpenRouterChatProvider
from hontology.retrieve import embed

# Readers live in judge.parse; these names are kept here for existing callers.
__all__ = [
    "AGGREGATIONS",
    "JudgeStats",
    "aggregate",
    "get_provider",
    "judge_run",
    "liveness",
    "parse_batch",
    "parse_verdict",
]

log = logging.getLogger(__name__)


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
    batch = {"candidates": candidates, "limit": limit, "progress": progress}

    if template.mode in (prompts.EXTRACT, prompts.HIERARCHICAL):
        run = session.get(Run, run_id)
        assert run is not None
        if template.mode == prompts.HIERARCHICAL:
            return judge_hierarchical(judging, ontology_id=run.ontology_id, **batch)
        embedder = None
        if template.leaf_top_k is not None:
            embed_provider = config.get("candidates", {}).get("embed_provider", "ollama")
            embedder = embed.get_provider(embed_provider)
        return judge_extract(judging, ontology_id=run.ontology_id, embedder=embedder, **batch)

    done = _already_judged(session, run_id)
    if template.mode == prompts.PER_DOCUMENT:
        return judge_batched(judging, done=done, **batch)
    return _judge_pairs(judging, done=done, **batch)


def _judge_pairs(
    judging: Judging,
    *,
    candidates: list[Candidate],
    done: set[tuple[int, int]],
    limit: int | None,
    progress: object | None,
) -> dict:
    """One call per pair, sampled and aggregated as the config says."""
    session, template, stats = judging.session, judging.template, judging.stats
    samples = max(1, judging.judge_config["samples"])
    processed = 0
    for candidate in candidates:
        key = (candidate.document_id, candidate.concept_id)
        if key in done:
            stats.skipped += 1
            continue
        if limit is not None and processed >= limit:
            break

        document = session.get(Document, candidate.document_id)
        concept = session.get(Concept, candidate.concept_id)
        if document is None or concept is None:
            continue

        body = judging.body(document)
        if not body:
            # No usable text: record it rather than silently dropping the pair,
            # so the denominator stays honest.
            session.add(
                judging.verdict(document.id, concept.id, samples=samples, error=NO_BODY)
            )
            stats.errors += 1
            processed += 1
            continue

        prompt = template.build_pair(document, concept, body, judging.body_limit)
        parsed_samples: list[dict] = []
        reasoning = ""
        latency = 0.0
        tokens_in = 0
        tokens_out = 0
        error = ""

        for _ in range(samples):
            try:
                completion = judging.ask(template.system, prompt)
                parsed_samples.append(parse_verdict(completion.text))
                reasoning = completion.reasoning or reasoning
                latency += completion.latency_s
                tokens_in += completion.input_tokens or 0
                tokens_out += completion.output_tokens or 0
                stats.spent(completion)
            except (ProviderError, json.JSONDecodeError) as exc:
                # One bad pair must not end the run.
                error = f"{type(exc).__name__}: {exc}"
                break

        cost = {"latency_s": latency, "input_tokens": tokens_in, "output_tokens": tokens_out}
        if parsed_samples and not error:
            result = aggregate(
                parsed_samples, judging.judge_config.get("aggregation", "majority")
            )
            session.add(
                judging.verdict(
                    document.id,
                    concept.id,
                    samples=samples,
                    matched=result["matched"],
                    confidence=result["confidence"],
                    vote_fraction=result["vote_fraction"],
                    locus_id=judging.iso2_to_locus.get(result["country"]),
                    evidence=result["evidence"] or None,
                    reasoning=reasoning or None,
                    **cost,
                )
            )
            stats.judged += 1
            stats.matched += int(result["matched"])
        else:
            session.add(
                judging.verdict(
                    document.id,
                    concept.id,
                    samples=samples,
                    error=error or "no samples parsed",
                    **cost,
                )
            )
            stats.errors += 1

        processed += 1
        # Update progress BEFORE committing, so the new value is part of the same
        # transaction as the verdict. Setting it afterwards leaves it uncommitted
        # until the next pair, and the final value is lost entirely — which makes
        # a poller watching this run appear to stall one pair short of done.
        if callable(progress):
            progress(processed, len(candidates))

        # Commit as we go: this is what makes a killed run resumable rather than
        # a total loss.
        session.commit()

    log.info("judge: %s", stats.as_dict())
    return stats.as_dict()


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
