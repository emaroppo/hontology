"""Versioned prompt templates.

A ``prompt_id`` bundles everything about *how* the model is asked: the system
guidance, the response shape, and the construction mode (one call per pair versus
one call per document). Two templates differing only in construction mode are two
prompt ids, because they produce different results and must be comparable as
separate experiments.

The id lands in the run's judge key and is stamped on every verdict, so a result
can always be traced to the exact wording that produced it. Templates are
append-only: editing one in place would silently change what every verdict
already recorded under its id means. So each released id is pinned to the
fingerprint of its rendered text in prompts.lock.json; judging refuses an id
whose wording has drifted from its pin, and the tests refuse an unpinned one.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from hontology.db.models import Concept, Document

# The wording lives in prompts.toml; this module only assembles it.
_WORDING = tomllib.loads(Path(__file__).with_name("prompts.toml").read_text(encoding="utf-8"))
_SYSTEM, _SHAPE, _TEXT = _WORDING["system"], _WORDING["shape"], _WORDING["text"]
# Each released prompt id's fingerprint (runs.versions.prompt_fingerprint).
# Judging refuses a pinned id whose wording no longer matches; see judge.run.
PINS: dict[str, str] = json.loads(
    Path(__file__).with_name("prompts.lock.json").read_text(encoding="utf-8")
)

DEFAULT_PROMPT_ID = "strict_v1"

PER_PAIR = "per-pair"
PER_DOCUMENT = "per-document"
# Top-down over a class hierarchy: one batched call per sibling set, descending
# only into the children of classes judged positive.
HIERARCHICAL = "hierarchical"
# Extract an article's events, then classify each event down the hierarchy.
EXTRACT = "extract"
# Modes that judge top-down through the hierarchy, gated on any selected leaf.
TOP_DOWN_MODES = (HIERARCHICAL, EXTRACT)

MAX_EVENTS: int = _WORDING["max_events"]
RESPONSE_SHAPE: str = _SHAPE["pair"]
# At the top level an event may also be "other": none of the classes.
OTHER_ID = 0


@dataclass(frozen=True)
class PromptTemplate:
    """How the model is asked, including how many pairs per call.

    ``mode`` is not decoration: a per-document template judges every candidate
    concept for one article in a single call, which is a different experiment
    from N separate calls and must be comparable as one. Templates declaring
    ``per-document`` must supply ``build_batch``.
    """

    prompt_id: str
    mode: str  # "per-pair" | "per-document"
    system: str
    build_pair: Callable[[Document, Concept, str, int], str]
    build_batch: Callable[[Document, list[Concept], str, int], str] | None = None
    # Hierarchical only: how parent classes are asked, when they are asked as
    # routing questions in calls of their own rather than as concepts.
    route_system: str | None = None
    build_route: Callable[[Document, list[Concept], str, int], str] | None = None
    # Extract only: listing an article's events, routing one event, and choosing
    # at most one leaf for it.
    extract_system: str | None = None
    build_extract: Callable[[Document, list[Concept], str, int], str] | None = None
    event_top_system: str | None = None
    event_route_system: str | None = None
    build_event_route: Callable[..., str] | None = None
    choose_system: str | None = None
    build_choose: Callable[[dict, list[Concept]], str] | None = None
    # Extract only: below the top level, offer each event the leaves most
    # similar to it under this embedding model instead of routing it further.
    leaf_embed_model: str | None = None
    leaf_top_k: int | None = None

    def __post_init__(self) -> None:
        if self.mode in (PER_DOCUMENT, HIERARCHICAL) and self.build_batch is None:
            raise ValueError(
                f"prompt {self.prompt_id!r} declares mode {self.mode!r} but "
                "supplies no build_batch"
            )
        if self.mode == EXTRACT and None in (
            self.extract_system,
            self.build_extract,
            self.event_top_system,
            self.event_route_system,
            self.build_event_route,
            self.choose_system,
            self.build_choose,
        ):
            raise ValueError(f"prompt {self.prompt_id!r}: extract mode needs all its builders")
        if self.mode not in (PER_PAIR, PER_DOCUMENT, HIERARCHICAL, EXTRACT):
            raise ValueError(f"prompt {self.prompt_id!r} has unknown mode {self.mode!r}")
        if (self.route_system is None) != (self.build_route is None):
            raise ValueError(
                f"prompt {self.prompt_id!r} needs both route_system and build_route"
            )
        if (self.leaf_embed_model is None) != (self.leaf_top_k is None):
            raise ValueError(
                f"prompt {self.prompt_id!r} needs both leaf_embed_model and leaf_top_k"
            )
        if self.leaf_top_k is not None and self.mode != EXTRACT:
            raise ValueError(
                f"prompt {self.prompt_id!r}: ranking leaves by embedding needs mode {EXTRACT!r}"
            )
        if self.build_route is not None and self.mode != HIERARCHICAL:
            raise ValueError(
                f"prompt {self.prompt_id!r}: routing questions need mode {HIERARCHICAL!r}"
            )


_REGISTRY: dict[str, PromptTemplate] = {}


def register(template: PromptTemplate) -> None:
    _REGISTRY[template.prompt_id] = template


def get(prompt_id: str) -> PromptTemplate:
    try:
        return _REGISTRY[prompt_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"no prompt template {prompt_id!r} (registered: {known})") from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _article(document: Document, body: str, body_limit: int) -> str:
    return (
        f"=== Article ===\n"
        f"url: {document.url}\n"
        f"title: {document.title or ''}\n"
        f"body: {_trim(body, body_limit)}\n\n"
    )


def _questions(entries) -> str:
    """Routing questions, as ``(concept_id, question)`` pairs."""
    return "\n\n".join(f"[concept_id {i}]\nquestion: {q}" for i, q in entries)


def _concept_block(concept: Concept) -> str:
    parts = [f"name: {concept.name}"]
    if concept.definition:
        parts.append(f"definition: {concept.definition}")
    # The criteria are the user's boundary-drawing, so they go in verbatim rather
    # than being summarized into the definition.
    if concept.inclusion_criteria:
        parts.append(f"counts when: {concept.inclusion_criteria}")
    if concept.exclusion_criteria:
        parts.append(f"does NOT count when: {concept.exclusion_criteria}")
    return "\n".join(parts)


def _event_block(event: dict) -> str:
    return (
        f"description: {event.get('description', '')}\n"
        f"status: {event.get('status', '')}\n"
        f"country: {event.get('country', '')}\n"
        f"evidence: {event.get('evidence', '')}"
    )


# ===========================================================================
# Builders
# ===========================================================================


def _build_strict(document: Document, concept: Concept, body: str, body_limit: int) -> str:
    return (
        _article(document, body, body_limit)
        + f"=== Concept ===\n{_concept_block(concept)}\n\n{RESPONSE_SHAPE}\n"
    )


def _build_concept_first(
    document: Document, concept: Concept, body: str, body_limit: int
) -> str:
    return (
        f"=== Concept ===\n{_concept_block(concept)}\n\n"
        + _article(document, body, body_limit)
        + f"{RESPONSE_SHAPE}\n"
    )


def _build_batch_strict(
    document: Document, concepts: list[Concept], body: str, body_limit: int
) -> str:
    blocks = "\n\n".join(
        f"[concept_id {concept.id}]\n{_concept_block(concept)}" for concept in concepts
    )
    return (
        _article(document, body, body_limit) + f"{_TEXT['batch'].format(n=len(concepts))}\n\n"
        f"=== Concepts ===\n{blocks}\n\n"
        f"{_SHAPE['batch']}\n"
    )


def _build_route(
    document: Document, concepts: list[Concept], body: str, body_limit: int
) -> str:
    # A parent's text is its routing question, held as ontology content.
    blocks = _questions((c.id, c.definition or c.name) for c in concepts)
    return (
        _article(document, body, body_limit) + f"{_TEXT['route'].format(n=len(concepts))}\n\n"
        f"=== Questions ===\n{blocks}\n\n"
        f"{_SHAPE['batch'].replace('concept listed', 'question listed')}\n"
    )


def _build_extract(document: Document, top: list[Concept], body: str, body_limit: int) -> str:
    # The top-level classes say what is in scope, so the list stays on topic.
    kinds = "\n".join(f"- {c.name}: {c.definition or c.name}" for c in top)
    return (
        _article(document, body, body_limit)
        + f"=== Kinds of event in scope ===\n{kinds}\n\n{_SHAPE['extract']}\n"
    )


def _build_event_route(event: dict, concepts: list[Concept], other: bool = False) -> str:
    entries = [(c.id, c.definition or c.name) for c in concepts]
    if other:
        entries.append((OTHER_ID, _TEXT["other_question"]))
    return (
        f"=== Event ===\n{_event_block(event)}\n\n"
        f"{_TEXT['event_route'].format(n=len(entries))}\n\n"
        f"=== Questions ===\n{_questions(entries)}\n\n"
        f"{_SHAPE['event_route']}\n"
    )


def _build_choose(event: dict, concepts: list[Concept]) -> str:
    blocks = "\n\n".join(f"[concept_id {c.id}]\n{_concept_block(c)}" for c in concepts)
    return (
        f"=== Event ===\n{_event_block(event)}\n\n"
        f"=== Concepts: choose at most one ===\n{blocks}\n\n"
        f"{_SHAPE['choose']}\n"
    )


# ===========================================================================
# Registered prompt ids. Why each wording is what it is: see prompts.toml.
# ===========================================================================

register(PromptTemplate("strict_v1", PER_PAIR, _SYSTEM["strict"], _build_strict))

# Identical guidance and response shape; only the ORDER differs. Putting the
# article first makes it a stable prefix across every concept tested against that
# document, which is what lets a server reuse the prompt's KV cache between the
# N calls for one article. strict_v1 puts the article first for exactly that
# reason; this variant puts the concept first so the effect can be measured
# rather than assumed.
register(PromptTemplate("concept_first_v1", PER_PAIR, _SYSTEM["strict"], _build_concept_first))

register(PromptTemplate("lenient_v1", PER_PAIR, _SYSTEM["lenient"], _build_strict))

# One call per document, judging every candidate at once, with strict_v1's
# guidance so the only difference is construction: far fewer calls and one copy
# of the body, against a longer prompt the model can get partially wrong.
# Self-consistency sampling does not apply: repeating a batch call re-rolls every
# verdict together, so the votes are not independent.
register(
    PromptTemplate(
        "strict_batch_v1",
        PER_DOCUMENT,
        _SYSTEM["strict"],
        _build_strict,
        build_batch=_build_batch_strict,
    )
)

# The hierarchical arm is asked exactly as the batched baseline is: the same
# system prompt, the same builder, the same response shape. Only which classes
# go into each call differs, so a difference between the arms is the hierarchy's
# and not the wording's.
register(replace(get("strict_batch_v1"), prompt_id="hier_batch_v1", mode=HIERARCHICAL))

# Parents asked as routing questions, leaning to yes, in calls of their own;
# leaves asked exactly as in hier_batch_v1 and the flat baseline, in separate
# calls, so any difference between the arms comes from the hierarchy.
register(
    replace(
        get("hier_batch_v1"),
        prompt_id="hier_batch_v2",
        route_system=_SYSTEM["route"],
        build_route=_build_route,
    )
)

# Arm E. A hierarchical judge asked about a whole article ticks several sibling
# classes for one event (a pipeline strike as a shutdown, a production halt and
# an industrial accident). Here the article's events are listed first, once,
# and each event is then routed down the hierarchy and given at most one leaf:
# one event, one class, any number of events. The article is read once; the
# routing and choosing calls see only the event's description and evidence.
# Leaf definitions are unchanged.
register(
    PromptTemplate(
        "extract_v1",
        EXTRACT,
        _SYSTEM["strict"],
        _build_strict,
        extract_system=_SYSTEM["extract"].replace("{max_events}", str(MAX_EVENTS)),
        build_extract=_build_extract,
        event_top_system=_SYSTEM["event_top"],
        event_route_system=_SYSTEM["event_route"],
        build_event_route=_build_event_route,
        choose_system=_SYSTEM["choose"],
        build_choose=_build_choose,
    )
)

# extract_v1 with the routing below the top level replaced. An event the top
# level accepts is offered the leaves under the classes it answered yes to that
# sit closest to it in embedding space, and nothing more is asked before the
# choosing call. Every text the judge reads is extract_v1's.
register(
    replace(
        get("extract_v1"),
        prompt_id="extract_embed_v1",
        leaf_embed_model="mxbai-embed-large",
        leaf_top_k=3,
    )
)
