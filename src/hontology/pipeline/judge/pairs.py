"""Per-pair judging: one call per ``(document, concept)``, sampled and aggregated."""

from __future__ import annotations

import json
import logging

from hontology.db.models import Candidate, Concept, Document
from hontology.pipeline.judge.context import NO_BODY, Judging
from hontology.pipeline.judge.parse import aggregate, parse_verdict
from hontology.pipeline.judge.providers.base import ProviderError

log = logging.getLogger(__name__)


def judge_pairs(
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
        parsed_samples, reasoning, cost, error = _sample(judging, prompt, samples)
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


def _sample(judging: Judging, prompt: str, samples: int) -> tuple[list[dict], str, dict, str]:
    """Ask *samples* times: the parsed replies, reasoning, summed cost and any error."""
    parsed_samples: list[dict] = []
    reasoning = ""
    latency = 0.0
    tokens_in = 0
    tokens_out = 0
    error = ""

    for _ in range(samples):
        try:
            completion = judging.ask(judging.template.system, prompt)
            parsed_samples.append(parse_verdict(completion.text))
            reasoning = completion.reasoning or reasoning
            latency += completion.latency_s
            tokens_in += completion.input_tokens or 0
            tokens_out += completion.output_tokens or 0
            judging.stats.spent(completion)
        except (ProviderError, json.JSONDecodeError) as exc:
            # One bad pair must not end the run.
            error = f"{type(exc).__name__}: {exc}"
            break

    cost = {"latency_s": latency, "input_tokens": tokens_in, "output_tokens": tokens_out}
    return parsed_samples, reasoning, cost, error
