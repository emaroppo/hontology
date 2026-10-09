"""The judge's wording: prompts.toml loaded, and the builders that assemble it.

Every word the judge reads that is not the article or the ontology is in
prompts.toml; the builders here put it together with an article, classes and
events. Which prompt id uses which builder is in `pipeline.judge.prompts`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from hontology.db.models import Concept, Document

_WORDING = tomllib.loads(Path(__file__).with_name("prompts.toml").read_text(encoding="utf-8"))
SYSTEM, SHAPE, TEXT = _WORDING["system"], _WORDING["shape"], _WORDING["text"]

MAX_EVENTS: int = _WORDING["max_events"]
RESPONSE_SHAPE: str = SHAPE["pair"]
# At the top level an event may also be "other": none of the classes.
OTHER_ID = 0


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


def build_strict(document: Document, concept: Concept, body: str, body_limit: int) -> str:
    return (
        _article(document, body, body_limit)
        + f"=== Concept ===\n{_concept_block(concept)}\n\n{RESPONSE_SHAPE}\n"
    )


def build_concept_first(
    document: Document, concept: Concept, body: str, body_limit: int
) -> str:
    return (
        f"=== Concept ===\n{_concept_block(concept)}\n\n"
        + _article(document, body, body_limit)
        + f"{RESPONSE_SHAPE}\n"
    )


def build_batch_strict(
    document: Document, concepts: list[Concept], body: str, body_limit: int
) -> str:
    blocks = "\n\n".join(
        f"[concept_id {concept.id}]\n{_concept_block(concept)}" for concept in concepts
    )
    return (
        _article(document, body, body_limit) + f"{TEXT['batch'].format(n=len(concepts))}\n\n"
        f"=== Concepts ===\n{blocks}\n\n"
        f"{SHAPE['batch']}\n"
    )


def build_route(document: Document, concepts: list[Concept], body: str, body_limit: int) -> str:
    # A parent's text is its routing question, held as ontology content.
    blocks = _questions((c.id, c.definition or c.name) for c in concepts)
    return (
        _article(document, body, body_limit) + f"{TEXT['route'].format(n=len(concepts))}\n\n"
        f"=== Questions ===\n{blocks}\n\n"
        f"{SHAPE['batch'].replace('concept listed', 'question listed')}\n"
    )


def build_extract(document: Document, top: list[Concept], body: str, body_limit: int) -> str:
    # The top-level classes say what is in scope, so the list stays on topic.
    kinds = "\n".join(f"- {c.name}: {c.definition or c.name}" for c in top)
    return (
        _article(document, body, body_limit)
        + f"=== Kinds of event in scope ===\n{kinds}\n\n{SHAPE['extract']}\n"
    )


def build_event_route(event: dict, concepts: list[Concept], other: bool = False) -> str:
    entries = [(c.id, c.definition or c.name) for c in concepts]
    if other:
        entries.append((OTHER_ID, TEXT["other_question"]))
    return (
        f"=== Event ===\n{_event_block(event)}\n\n"
        f"{TEXT['event_route'].format(n=len(entries))}\n\n"
        f"=== Questions ===\n{_questions(entries)}\n\n"
        f"{SHAPE['event_route']}\n"
    )


def build_choose(event: dict, concepts: list[Concept]) -> str:
    blocks = "\n\n".join(f"[concept_id {c.id}]\n{_concept_block(c)}" for c in concepts)
    return (
        f"=== Event ===\n{_event_block(event)}\n\n"
        f"=== Concepts: choose at most one ===\n{blocks}\n\n"
        f"{SHAPE['choose']}\n"
    )
