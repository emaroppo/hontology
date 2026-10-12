"""What every retrieval strategy shares: an article in, offers for the judge out.

A strategy is a list of steps run in order over one `Retrieval`. Each step reads
what earlier steps left there, adds its own part, and writes a line to the
trace. Steps vary between strategies; the ends do not:

- **In:** an `Article`, a document and the body the judge would read.
- **Out:** `Retrieval.offers`, the leaves each deciding call will be shown.
  Those leaves, and only those, are what the article *retrieved*.

A step that calls the model asks through a `Model`, which a judge run and a
try-out both provide. No step writes to the database: a run records what a
retrieval found, and a try-out shows it and drops it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from hontology.db.models import Concept, Document
from hontology.pipeline.judge.context import NO_BODY
from hontology.pipeline.judge.prompts import PromptTemplate
from hontology.pipeline.judge.providers.base import Completion, ProviderError

# What a model call may fail with; a step records it rather than raising.
CALL_ERRORS = (ProviderError, json.JSONDecodeError)


class Model(Protocol):
    """How a step asks the judge model, and reads the classes it asks about."""

    template: PromptTemplate
    body_limit: int

    def ask(self, system: str | None, prompt: str) -> Completion: ...

    def concepts(self, ids: Iterable[int]) -> list[Concept]: ...


@dataclass
class Article:
    document: Document
    # Cut to the judge's length; None when there is no usable text.
    body: str | None


class Spend:
    """What a retrieval's model calls cost, and how many there were."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0
        self.calls = 0

    def add(self, completion: Completion) -> Completion:
        self.input_tokens += completion.input_tokens or 0
        self.output_tokens += completion.output_tokens or 0
        self.seconds += completion.latency_s or 0.0
        self.calls += 1
        return completion


@dataclass
class Asked:
    """One routing call over classes, with each class's parsed answer.

    ``replies`` is None when the call failed, and ``error`` says why.
    """

    concept_ids: list[int]
    replies: dict[int, dict] | None
    completion: Completion | None
    error: str | None = None

    def answer(self, concept_id: int) -> bool | None:
        return None if self.replies is None else bool(self.replies[concept_id]["matched"])


@dataclass
class Offer:
    """The leaves one deciding call is shown, and the event it is about, if any."""

    concept_ids: list[int]
    event: int | None = None


@dataclass
class Retrieval:
    """One article's retrieval, as its steps build it up."""

    article: Article
    offers: list[Offer] = field(default_factory=list)
    # Routing calls over parent classes, in the order they were made.
    asked: list[Asked] = field(default_factory=list)
    # Events found in the article, for strategies that extract them.
    events: list = field(default_factory=list)
    # Each parent class: answered yes for any event (event strategies).
    parent_yes: dict[int, bool] = field(default_factory=dict)
    spend: Spend = field(default_factory=Spend)
    trace: list[dict] = field(default_factory=list)
    # Why retrieval stopped before its last step: a closed gate, no text, or a
    # failure that leaves nothing to offer.
    stopped: str | None = None
    error: str | None = None

    @property
    def shown(self) -> set[int]:
        """Every leaf put in front of a deciding call: what was retrieved."""
        return {c for offer in self.offers for c in offer.concept_ids}

    def note(self, step: str, **details) -> None:
        self.trace.append({"step": step, **details})


class Step(Protocol):
    name: str

    def __call__(self, retrieval: Retrieval) -> None: ...


@dataclass
class Strategy:
    """A named list of steps, run in order until one stops the retrieval."""

    name: str
    steps: list[Step]

    def retrieve(self, article: Article) -> Retrieval:
        retrieval = Retrieval(article)
        if not article.body:
            retrieval.stopped = retrieval.error = NO_BODY
            return retrieval
        for step in self.steps:
            step(retrieval)
            if retrieval.stopped:
                break
        return retrieval
