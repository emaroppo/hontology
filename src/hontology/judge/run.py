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
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Candidate, Concept, Document, Locus, Verdict
from hontology.judge import prompts
from hontology.judge.providers.base import GenerationConfig, ProviderError
from hontology.judge.providers.llamacpp import LlamaCppChatProvider
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
    if name == "llamacpp":
        return LlamaCppChatProvider(settings.llamacpp_host, settings.llamacpp_api_key)
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

    return _coerce(payload)


AGGREGATIONS = ("majority", "unanimous", "any")


def aggregate(samples: list[dict], how: str = "majority") -> dict:
    """Combine repeated samples into one verdict.

    The vote fraction replaces the model's self-reported confidence, because it
    is the more honest signal: a model asked five times and answering yes three
    times is genuinely uncertain in a way its own stated 0.9 does not capture.

    The three rules trade precision against recall explicitly:

    - ``majority`` — the plurality answer. Balanced, and the default.
    - ``unanimous`` — matched only if *every* sample says so. Fewer, surer
      matches; the choice when a false positive is expensive.
    - ``any`` — matched if *any* sample says so. Catches concepts the model
      only occasionally notices, at the cost of precision.

    The vote fraction is reported unchanged under all three, so a verdict's
    uncertainty stays visible however the rule resolved it.
    """
    if how not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {how!r}; expected one of {AGGREGATIONS}")

    if len(samples) == 1:
        return samples[0] | {"vote_fraction": 1.0}

    positives = sum(1 for s in samples if bool(s["matched"]))
    if how == "unanimous":
        winner = positives == len(samples)
    elif how == "any":
        winner = positives > 0
    else:
        winner = positives * 2 > len(samples)

    # Always the share that agreed with the *reported* verdict.
    agreeing = positives if winner else len(samples) - positives
    fraction = agreeing / len(samples)
    representative = next((s for s in samples if bool(s["matched"]) == winner), samples[0])

    return representative | {
        "matched": winner,
        "confidence": fraction,
        "vote_fraction": fraction,
    }


def parse_batch(raw: str, expected_concept_ids: list[int]) -> dict[int, dict]:
    """Parse one batched response into ``{concept_id: verdict}``.

    Every requested concept gets an entry. A model that omits one would
    otherwise silently shrink the denominator — the pair would simply vanish
    rather than count as a miss — so anything absent is filled with a
    zero-confidence non-match and the omission is visible in the counts.
    """
    payload = json.loads(raw) if raw.strip().startswith("{") else None
    if payload is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise json.JSONDecodeError("no JSON object found", raw, 0)
        payload = json.loads(raw[start : end + 1])

    entries = payload.get("verdicts")
    if not isinstance(entries, list):
        raise json.JSONDecodeError("no 'verdicts' array in response", raw, 0)

    by_concept: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("concept_id")
        if raw_id is None:
            continue
        try:
            concept_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if concept_id not in expected_concept_ids:
            # The model invented a concept; ignore rather than store a row that
            # references nothing.
            continue
        by_concept[concept_id] = _coerce(entry) | {"omitted": False}

    for concept_id in expected_concept_ids:
        by_concept.setdefault(
            concept_id,
            {
                "matched": False,
                "confidence": 0.0,
                "country": "",
                "evidence": "",
                "omitted": True,
            },
        )
    return by_concept


def _coerce(payload: dict) -> dict:
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
    provider = get_provider(judge_config["provider"])
    model = judge_config["model"]
    samples = max(1, judge_config["samples"])

    generation = GenerationConfig(
        temperature=judge_config["generation"]["temperature"],
        context_window=judge_config["generation"]["context_window"],
        max_output_tokens=judge_config["generation"]["max_output_tokens"],
        seed=judge_config["generation"]["seed"],
    )

    query = select(Candidate).where(Candidate.run_id == run_id, Candidate.selected.is_(True))
    if document_ids is not None:
        query = query.where(Candidate.document_id.in_(document_ids))
    order = (
        (Candidate.score.desc(), Candidate.document_id, Candidate.concept_id)
        if by_score
        else (Candidate.document_id, Candidate.concept_id)
    )
    candidates = list(session.scalars(query.order_by(*order)))
    done = _already_judged(session, run_id)

    stats = JudgeStats(total=len(candidates))
    iso2_to_locus: dict[str, int] = {
        locus.iso2: locus.id
        for locus in session.scalars(select(Locus).where(Locus.iso2.is_not(None)))
        if locus.iso2
    }

    if template.mode == prompts.PER_DOCUMENT:
        return _judge_batched(
            session,
            run_id,
            candidates=candidates,
            done=done,
            stats=stats,
            template=template,
            provider=provider,
            judge_config=judge_config,
            generation=generation,
            judge_body_limit=judge_body_limit,
            iso2_to_locus=iso2_to_locus,
            limit=limit,
            progress=progress,
        )

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
            result = aggregate(parsed_samples, judge_config.get("aggregation", "majority"))
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


def _judge_batched(
    session: Session,
    run_id: int,
    *,
    candidates: list[Candidate],
    done: set[tuple[int, int]],
    stats: JudgeStats,
    template: prompts.PromptTemplate,
    provider,
    judge_config: dict,
    generation: GenerationConfig,
    judge_body_limit: int,
    iso2_to_locus: dict[str, int],
    limit: int | None,
    progress: object | None,
) -> dict:
    """One call per document, judging every candidate concept at once.

    The cost profile is the point: N concepts for one article cost one call and
    one copy of the body, rather than N calls each re-sending it. What is traded
    away is isolation — a single malformed response costs every pair for that
    document, not one — so a failure is recorded against each of them rather than
    losing them silently.

    Sampling does not apply. Repeating a batch call re-rolls every verdict
    together, so the votes are not independent and a vote fraction across them
    would overstate agreement. Batch runs are a single greedy call.
    """
    by_document: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        if (candidate.document_id, candidate.concept_id) in done:
            stats.skipped += 1
            continue
        by_document.setdefault(candidate.document_id, []).append(candidate)

    processed = 0
    omitted_total = 0

    for document_id, group in by_document.items():
        if limit is not None and processed >= limit:
            break

        document = session.get(Document, document_id)
        if document is None:
            continue
        concepts: list[Concept] = [
            found
            for found in (session.get(Concept, c.concept_id) for c in group)
            if found is not None
        ]
        if not concepts:
            continue

        body = _body(document, judge_body_limit)
        common = {
            "run_id": run_id,
            "provider": judge_config["provider"],
            "model": judge_config["model"],
            "prompt_id": template.prompt_id,
            "mode": template.mode,
            "samples": 1,
        }

        if not body:
            for concept in concepts:
                session.add(
                    Verdict(
                        document_id=document_id,
                        concept_id=concept.id,
                        error="no article body available",
                        **common,
                    )
                )
                stats.errors += 1
            processed += len(concepts)
            session.commit()
            continue

        assert template.build_batch is not None  # guaranteed by PromptTemplate
        prompt = template.build_batch(document, concepts, body, judge_body_limit)

        try:
            completion = provider.complete(
                system=template.system,
                prompt=prompt,
                config=generation,
                want_json=True,
                want_reasoning=judge_config["think"],
                model=judge_config["model"],
            )
            parsed = parse_batch(completion.text, [c.id for c in concepts])
            stats.input_tokens += completion.input_tokens or 0
            stats.output_tokens += completion.output_tokens or 0
        except (ProviderError, json.JSONDecodeError) as exc:
            # One bad response costs the whole document — recorded per pair so
            # the denominator stays honest and liveness catches it.
            for concept in concepts:
                session.add(
                    Verdict(
                        document_id=document_id,
                        concept_id=concept.id,
                        error=f"{type(exc).__name__}: {exc}",
                        **common,
                    )
                )
                stats.errors += 1
            processed += len(concepts)
            session.commit()
            continue

        for concept in concepts:
            result = parsed[concept.id]
            if result.get("omitted"):
                omitted_total += 1
            session.add(
                Verdict(
                    document_id=document_id,
                    concept_id=concept.id,
                    matched=result["matched"],
                    confidence=result["confidence"],
                    vote_fraction=1.0,
                    locus_id=iso2_to_locus.get(result["country"]),
                    evidence=result["evidence"] or None,
                    reasoning=completion.reasoning or None,
                    # One call served the whole group, so attributing its full
                    # latency to each pair would multiply the real cost.
                    latency_s=completion.latency_s / len(concepts),
                    **common,
                )
            )
            stats.judged += 1
            stats.matched += int(result["matched"])

        processed += len(concepts)
        if callable(progress):
            progress(processed, len(candidates))
        session.commit()

    result = stats.as_dict() | {
        "mode": template.mode,
        "documents": len(by_document),
        # Concepts the model left out of its array. They count as non-matches so
        # the denominator holds, but a high number means the batch prompt is
        # overloaded and the model is dropping items.
        "omitted_by_model": omitted_total,
    }
    log.info("judge (batched): %s", result)
    return result
