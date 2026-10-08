"""Asking the judge about one pair, with a prompt and class wording of choice.

The Judgement page edits a prompt and sees what the model would say. That needs
the exact path a run takes, minus everything a run records: the same template
builder, the same provider, model and decoding settings as a chosen run, the
same article text limit. Nothing here writes a verdict, so a trial can never be
mistaken for a run's result.

Only per-pair templates can be tried this way. Their prompt is one system text
and one message built from the article and the class's wording, which is the
unit a person can sensibly edit; batched, hierarchical and extraction prompts
are several calls whose shape depends on earlier answers.

Edited class wording is a detached copy of the class, never added to the
session, so trying a wording cannot change the ontology or mint a version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Document, Run, Verdict
from hontology.evalkit import config as run_config
from hontology.judge import prompts
from hontology.judge.providers.base import GenerationConfig, ProviderError
from hontology.judge.run import _body, get_provider, parse_verdict

WORDING_FIELDS = ("definition", "inclusion_criteria", "exclusion_criteria")


@dataclass(frozen=True)
class Trial:
    run_id: int
    document_id: int
    concept_id: int
    prompt_id: str = prompts.DEFAULT_PROMPT_ID
    # None keeps the template's own system text and the class's saved wording.
    system: str | None = None
    wording: dict[str, str | None] | None = None


def per_pair_templates() -> list[dict]:
    return [
        {"prompt_id": prompt_id, "system": template.system}
        for prompt_id in prompts.available()
        if (template := prompts.get(prompt_id)).mode == prompts.PER_PAIR
    ]


def _judge_config(run: Run) -> tuple[dict, int]:
    config = run_config.normalize(run.config or {})
    return config["judge"], config["common"]["judge_body_limit"]


def render(session: Session, trial: Trial) -> dict:
    """The system text and message a trial would send."""
    run = session.get(Run, trial.run_id)
    document = session.get(Document, trial.document_id)
    concept = session.get(Concept, trial.concept_id)
    if run is None or document is None or concept is None:
        raise LookupError("unknown run, article or class")
    template = prompts.get(trial.prompt_id)
    if template.mode != prompts.PER_PAIR:
        raise ValueError(f"{trial.prompt_id!r} is not a per-pair prompt")

    asked = concept
    if trial.wording is not None:
        unknown = set(trial.wording) - set(WORDING_FIELDS)
        if unknown:
            raise ValueError(f"not wording fields: {', '.join(sorted(unknown))}")
        # Detached: never added to the session, so nothing is saved.
        asked = Concept(
            name=concept.name,
            **{
                field: (trial.wording.get(field, getattr(concept, field)) or None)
                for field in WORDING_FIELDS
            },
        )
    _, body_limit = _judge_config(run)
    body = _body(document, body_limit) or ""
    return {
        "system": trial.system if trial.system is not None else template.system,
        "prompt": template.build_pair(document, asked, body, body_limit),
        "has_body": bool(body),
    }


def ask(session: Session, trial: Trial) -> dict:
    """What the model says to a trial, asked as the run would ask it."""
    rendered = render(session, trial)
    run = session.get(Run, trial.run_id)
    assert run is not None
    judge_config, _ = _judge_config(run)
    generation = judge_config["generation"]
    provider = get_provider(judge_config["provider"], routing=judge_config.get("routing"))
    out = rendered | {
        "provider": judge_config["provider"],
        "model": judge_config["model"],
        "error": None,
    }
    try:
        completion = provider.complete(
            system=rendered["system"],
            prompt=rendered["prompt"],
            config=GenerationConfig(
                temperature=generation["temperature"],
                context_window=generation["context_window"],
                max_output_tokens=generation["max_output_tokens"],
                seed=generation["seed"],
            ),
            want_json=True,
            want_reasoning=judge_config["think"],
            model=judge_config["model"],
        )
    except ProviderError as exc:
        return out | {"error": str(exc)}
    out |= {
        "raw": completion.text,
        "reasoning": completion.reasoning or None,
        "latency_s": completion.latency_s,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
    }
    try:
        return out | parse_verdict(completion.text)
    except json.JSONDecodeError as exc:
        return out | {"error": f"unreadable reply: {exc}"}


def articles(
    session: Session,
    run_id: int,
    concept_id: int,
    labels: dict[tuple[int, int], bool],
    *,
    limit: int = 40,
) -> list[dict]:
    """Articles to try a class on, the most informative first.

    Each carries its label for this class and what the run said about it, so a
    trial's answer can be read against both.
    """
    labelled = {doc: matched for (doc, cid), matched in labels.items() if cid == concept_id}
    verdicts = {
        v.document_id: v
        for v in session.scalars(
            select(Verdict)
            .where(Verdict.run_id == run_id, Verdict.concept_id == concept_id)
            .order_by(Verdict.matched.desc().nulls_last(), Verdict.confidence.desc())
            .limit(limit)
        )
    }
    # Labelled ones the run also judged are worth having their verdict shown.
    for v in session.scalars(
        select(Verdict).where(
            Verdict.run_id == run_id,
            Verdict.concept_id == concept_id,
            Verdict.document_id.in_(list(labelled)),
        )
    ):
        verdicts[v.document_id] = v

    def priority(doc: int) -> tuple[int, int]:
        """Most informative first: true matches, then what the run said yes to,
        then labelled articles the run judged, then the rest."""
        label, verdict = labelled.get(doc), verdicts.get(doc)
        if label:
            return (0, doc)
        if verdict is not None and verdict.matched:
            return (1, doc)
        if label is not None and verdict is not None:
            return (2, doc)
        return (3 if verdict is not None else 4, doc)

    order = sorted(set(labelled) | set(verdicts), key=priority)[:limit]
    documents = {
        d.id: d for d in session.scalars(select(Document).where(Document.id.in_(order)))
    }
    out = []
    for doc in order:
        document = documents.get(doc)
        if document is None:
            continue
        verdict = verdicts.get(doc)
        out.append(
            {
                "document_id": doc,
                "url": document.url,
                "title": document.title,
                "label": labelled.get(doc),
                "verdict": None
                if verdict is None
                else {
                    "matched": verdict.matched,
                    "confidence": verdict.confidence,
                    "evidence": verdict.evidence,
                    "error": verdict.error,
                    "prompt_id": verdict.prompt_id,
                },
            }
        )
    return out
