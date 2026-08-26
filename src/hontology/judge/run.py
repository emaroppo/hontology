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
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Concept, Document, Locus, Verdict
from hontology.judge import prompts
from hontology.judge.providers.base import GenerationConfig, ProviderError
from hontology.judge.providers.ollama import OllamaChatProvider

log = logging.getLogger(__name__)


@dataclass
class JudgeStats:
    total: int = 0
    judged: int = 0
    skipped: int = 0
    matched: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "judged": self.judged,
            "skipped_already_done": self.skipped,
            "matched": self.matched,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def get_provider(name: str):
    settings = get_settings()
    if name == "ollama":
        return OllamaChatProvider(settings.ollama_host)
    raise ProviderError(f"unknown judge provider {name!r}")


def parse_verdict(raw: str) -> dict:
    """Parse a model response into the canonical fields.

    Falls back to the outermost ``{...}`` block: models reliably wrap JSON in
    prose or fences even when told not to, and discarding an otherwise good
    answer over a stray "Here you go:" would be throwing away real work.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "matched": bool(payload.get("matched", False)),
        "confidence": max(0.0, min(1.0, confidence)),
        "country": str(payload.get("country") or "").strip().upper(),
        "evidence": str(payload.get("evidence") or "").strip(),
    }


def aggregate(samples: list[dict]) -> dict:
    """Majority vote over repeated samples.

    The vote fraction replaces the model's self-reported confidence, because it
    is the more honest signal: a model asked five times and answering yes three
    times is genuinely uncertain in a way its own stated 0.9 does not capture.
    """
    if len(samples) == 1:
        return samples[0] | {"vote_fraction": 1.0}

    votes = Counter(bool(s["matched"]) for s in samples)
    winner, count = votes.most_common(1)[0]
    fraction = count / len(samples)
    representative = next(s for s in samples if bool(s["matched"]) == winner)

    return representative | {
        "matched": winner,
        "confidence": fraction,
        "vote_fraction": fraction,
    }


def _already_judged(session: Session, run_id: int) -> set[tuple[int, int]]:
    return {
        (document_id, concept_id)
        for document_id, concept_id in session.execute(
            select(Verdict.document_id, Verdict.concept_id).where(Verdict.run_id == run_id)
        )
    }


def _body(document: Document, limit: int) -> str | None:
    settings = get_settings()
    if not document.body_path:
        return None
    path = settings.scrape_cache_dir / document.body_path
    return path.read_text(encoding="utf-8")[:limit] if path.exists() else None


def judge_run(
    session: Session,
    run_id: int,
    *,
    config: dict,
    judge_body_limit: int,
    limit: int | None = None,
    progress: object | None = None,
) -> dict:
    """Judge every selected candidate for a run, skipping those already done."""
    judge_config = config["judge"]
    template = prompts.get(judge_config["prompt_id"])
    provider = get_provider(judge_config["provider"])
    model = judge_config["model"]
    samples = max(1, judge_config["samples"])

    generation = GenerationConfig(
        temperature=judge_config["generation"]["temperature"],
        context_window=judge_config["generation"]["context_window"],
        max_output_tokens=judge_config["generation"]["max_output_tokens"],
        seed=judge_config["generation"]["seed"],
    )

    candidates = list(
        session.scalars(
            select(Candidate)
            .where(Candidate.run_id == run_id, Candidate.selected.is_(True))
            .order_by(Candidate.document_id, Candidate.concept_id)
        )
    )
    done = _already_judged(session, run_id)

    stats = JudgeStats(total=len(candidates))
    iso2_to_locus = {
        locus.iso2: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso2.is_not(None)))
    }

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

        body = _body(document, judge_body_limit)
        if not body:
            # No usable text: record it rather than silently dropping the pair,
            # so the denominator stays honest.
            session.add(
                Verdict(
                    run_id=run_id,
                    document_id=document.id,
                    concept_id=concept.id,
                    provider=judge_config["provider"],
                    model=model,
                    prompt_id=template.prompt_id,
                    mode=template.mode,
                    samples=samples,
                    error="no article body available",
                )
            )
            stats.errors += 1
            processed += 1
            continue

        prompt = template.build_pair(document, concept, body, judge_body_limit)
        parsed_samples: list[dict] = []
        reasoning = ""
        latency = 0.0
        error = ""

        for _ in range(samples):
            try:
                completion = provider.complete(
                    system=template.system,
                    prompt=prompt,
                    config=generation,
                    want_json=True,
                    want_reasoning=judge_config["think"],
                    model=model,
                )
                parsed_samples.append(parse_verdict(completion.text))
                reasoning = completion.reasoning or reasoning
                latency += completion.latency_s
                stats.input_tokens += completion.input_tokens or 0
                stats.output_tokens += completion.output_tokens or 0
            except (ProviderError, json.JSONDecodeError) as exc:
                # One bad pair must not end the run.
                error = f"{type(exc).__name__}: {exc}"
                break

        if parsed_samples and not error:
            result = aggregate(parsed_samples)
            session.add(
                Verdict(
                    run_id=run_id,
                    document_id=document.id,
                    concept_id=concept.id,
                    matched=result["matched"],
                    confidence=result["confidence"],
                    vote_fraction=result["vote_fraction"],
                    locus_id=iso2_to_locus.get(result["country"]),
                    evidence=result["evidence"] or None,
                    reasoning=reasoning or None,
                    provider=judge_config["provider"],
                    model=model,
                    prompt_id=template.prompt_id,
                    mode=template.mode,
                    samples=samples,
                    latency_s=latency,
                )
            )
            stats.judged += 1
            stats.matched += int(result["matched"])
        else:
            session.add(
                Verdict(
                    run_id=run_id,
                    document_id=document.id,
                    concept_id=concept.id,
                    provider=judge_config["provider"],
                    model=model,
                    prompt_id=template.prompt_id,
                    mode=template.mode,
                    samples=samples,
                    latency_s=latency,
                    error=error or "no samples parsed",
                )
            )
            stats.errors += 1

        processed += 1
        # Commit as we go: this is what makes a killed run resumable rather than
        # a total loss.
        session.commit()

        if callable(progress):
            progress(processed, len(candidates))

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
