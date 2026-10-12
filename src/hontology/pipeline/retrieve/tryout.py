"""Trying a retrieval strategy on one article, or on text a person pastes in.

The strategy is the one a prompt template implies (see `strategies`), with the
chosen run's settings: its embedding version for the selection step, its judge
model, decoding settings and article length for any step that asks the model.
The selection step ranks live, through a cutoff of choice, so a try-out can
change the cutoff and see what routing or extraction does with the result.

Nothing is stored: no verdict, no event, no embedding of pasted text. Routing
and extraction ask the judge model, so unlike an embedding try-out they cost
model calls; the result says how many, and with which model.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from hontology.db.models import Concept, Document, Run
from hontology.ontology import hierarchy
from hontology.pipeline.judge import prompts
from hontology.pipeline.judge.providers.base import ChatProvider, Completion, GenerationConfig
from hontology.pipeline.retrieve import embed, pasted, strategies
from hontology.pipeline.retrieve.candidates import document_body
from hontology.pipeline.retrieve.strategies.selection import Selected
from hontology.pipeline.retrieve.tuning import Cutoff
from hontology.pipeline.runs import config as run_config

# The selection step ranks this much of a stored article, as retrieval would.
_RANK_LIMIT = 100_000


@dataclass
class TryModel:
    """A run's judge model, asked outside the run."""

    session: Session
    template: prompts.PromptTemplate
    provider: ChatProvider
    judge_config: dict
    body_limit: int

    def ask(self, system: str | None, prompt: str) -> Completion:
        return self.provider.complete(
            system=system,
            prompt=prompt,
            config=GenerationConfig.from_config(self.judge_config["generation"]),
            want_json=True,
            want_reasoning=self.judge_config["think"],
            model=self.judge_config["model"],
        )

    def concepts(self, ids) -> list[Concept]:
        found = (self.session.get(Concept, concept_id) for concept_id in ids)
        return [concept for concept in found if concept is not None]


def try_strategy(
    session: Session,
    run: Run,
    prompt_id: str,
    cutoff: Cutoff,
    *,
    text: str | None = None,
    document_id: int | None = None,
) -> dict:
    """Run *prompt_id*'s retrieval on one article or pasted text, and describe it."""
    from hontology.pipeline.judge.run import get_provider

    template = prompts.get(prompt_id)
    config = run_config.normalize(run.config or {})
    judge_config, body_limit = config["judge"], config["common"]["judge_body_limit"]
    if text is not None:
        if not text.strip():
            raise ValueError("there is no text to retrieve for")
        # Detached: never added to the session.
        document, full = Document(url="", title=None), text
    elif document_id is not None:
        found = session.get(Document, document_id)
        if found is None:
            raise LookupError(f"no article {document_id}")
        document, full = found, document_body(found, _RANK_LIMIT) or ""
    else:
        raise ValueError("give an article or some text")

    model = TryModel(
        session,
        template,
        get_provider(judge_config["provider"], routing=judge_config.get("routing")),
        judge_config,
        body_limit,
    )
    flat = template.mode in (prompts.PER_PAIR, prompts.PER_DOCUMENT)

    def choose(_: strategies.Article) -> list[int]:
        ranked = pasted.rank_text(session, run, full, cutoff)["ranking"]
        return [row["concept_id"] for row in ranked if row["kept"]]

    ranker = None
    if template.leaf_top_k is not None:
        assert template.leaf_embed_model is not None
        ranker = strategies.LeafRanker(
            embed.get_provider(config.get("candidates", {}).get("embed_provider", "ollama")),
            template.leaf_embed_model,
            model.concepts(sorted(hierarchy.leaves(session, run.ontology_id))),
            template.leaf_top_k,
        )
    strategy = strategies.for_template(
        model,
        top=sorted(hierarchy.top_level(session, run.ontology_id)),
        children=hierarchy.children(session, run.ontology_id),
        selected=Selected(choose, offer=flat),
        ranker=ranker,
    )
    retrieval = strategy.retrieve(strategies.Article(document, full[:body_limit] or None))
    return describe(model, strategy, retrieval) | {
        "prompt_id": prompt_id,
        "provider": judge_config["provider"],
        "model": judge_config["model"],
    }


def describe(model: TryModel, strategy: strategies.Strategy, retrieval) -> dict:
    """A retrieval as plain data, with class names for the page to show."""
    ids = {c for a in retrieval.asked for c in a.concept_ids} | retrieval.shown
    ids |= {c for e in retrieval.events for c in e.routed}
    names = {c.id: c.name for c in model.concepts(sorted(ids))}
    events = retrieval.events
    return {
        "strategy": strategy.name,
        "steps": [step.name for step in strategy.steps],
        "trace": retrieval.trace,
        "stopped": retrieval.stopped,
        "error": retrieval.error,
        "shown": sorted(names[c] for c in retrieval.shown),
        "offers": [
            {
                "event": None if o.event is None else events[o.event].found["description"],
                "classes": [names[c] for c in o.concept_ids],
            }
            for o in retrieval.offers
        ],
        "routing": [
            {
                "asked": [names[c] for c in a.concept_ids],
                "yes": [names[c] for c in a.concept_ids if a.answer(c)],
                "error": a.error,
            }
            for a in retrieval.asked
        ],
        "events": [
            {
                **{k: e.found.get(k) for k in ("description", "evidence", "status", "country")},
                "routed_through": [names[c] for c in e.routed],
                "offered": sorted(names[c] for c in e.reached) if not e.error else [],
                "error": e.error,
            }
            for e in events
        ],
        "spend": {
            "calls": retrieval.spend.calls,
            "input_tokens": retrieval.spend.input_tokens,
            "output_tokens": retrieval.spend.output_tokens,
            "seconds": retrieval.spend.seconds,
        },
    }
