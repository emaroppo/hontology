"""Versioned prompt templates.

A ``prompt_id`` bundles everything about *how* the model is asked: the system
guidance, the response shape, and the construction mode (one call per pair versus
one call per document). Two templates differing only in construction mode are two
prompt ids, because they produce different results and must be comparable as
separate experiments.

The id lands in the run's judge key and is stamped on every verdict, so a result
can always be traced to the exact wording that produced it. Templates are
append-only: editing one in place silently invalidates every verdict already
recorded under its id.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hontology.db.models import Concept, Document

DEFAULT_PROMPT_ID = "strict_v1"

PER_PAIR = "per-pair"
PER_DOCUMENT = "per-document"
# Top-down over a class hierarchy: one batched call per sibling set, descending
# only into the children of classes judged positive.
HIERARCHICAL = "hierarchical"

RESPONSE_SHAPE = (
    "Output format, a single JSON object:\n"
    '{"matched": <true|false>, "confidence": <0..1>, '
    '"country": "<two-letter ISO country code for where it happened; '
    'empty when matched is false>", '
    '"evidence": "<the exact passage copied from the article; '
    'empty when matched is false>"}'
)


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

    def __post_init__(self) -> None:
        if self.mode in (PER_DOCUMENT, HIERARCHICAL) and self.build_batch is None:
            raise ValueError(
                f"prompt {self.prompt_id!r} declares mode {self.mode!r} but "
                "supplies no build_batch"
            )
        if self.mode not in (PER_PAIR, PER_DOCUMENT, HIERARCHICAL):
            raise ValueError(f"prompt {self.prompt_id!r} has unknown mode {self.mode!r}")
        if (self.route_system is None) != (self.build_route is None):
            raise ValueError(
                f"prompt {self.prompt_id!r} needs both route_system and build_route"
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


# ===========================================================================
# strict_v1
#
# The exclusion list is the substance. A classifier told only "does this article
# evidence this concept" reliably says yes to articles about an event being
# *planned*, *demanded*, *feared*, or *averted* — all of which are about the
# concept without being instances of it. Naming those cases explicitly is the
# single biggest precision lever available before touching the model.
# ===========================================================================

_STRICT_SYSTEM = (
    "You decide whether a news article reports a real occurrence of a concept. "
    "A match needs the article to show that an instance of the concept has "
    "actually taken place; being about the concept is not enough.\n"
    "The concept's definition and its counts-when / does-not-count-when "
    "criteria are authoritative. Its name is only a label.\n"
    "Return matched=false in each of these cases:\n"
    "  NOT YET: the event is scheduled, anticipated, proposed, or announced as "
    "an intention.\n"
    "  ONLY URGED: someone calls for it, warns it may come, or threatens it.\n"
    "  FELL SHORT: it was attempted, but the outcome the concept requires did "
    "not follow.\n"
    "  DENIED: the article says it did not happen, or that it was prevented.\n"
    "  BACKGROUND: an earlier occurrence cited for context, a hypothetical, or "
    "an incidental mention.\n"
    "  NOISE: the relevant words appear only in menus, boilerplate or a "
    "paywall notice.\n"
    "Copy the evidence exactly from the article text; do not summarise it or "
    "make it up. Give confidence as how likely your answer is to be right, "
    "from 0 to 1.\n"
    "Output one JSON object and nothing else."
)


def _build_strict(document: Document, concept: Concept, body: str, body_limit: int) -> str:
    return (
        f"=== Article ===\n"
        f"url: {document.url}\n"
        f"title: {document.title or ''}\n"
        f"body: {_trim(body, body_limit)}\n\n"
        f"=== Concept ===\n{_concept_block(concept)}\n\n"
        f"{RESPONSE_SHAPE}\n"
    )


register(
    PromptTemplate(
        prompt_id="strict_v1",
        mode="per-pair",
        system=_STRICT_SYSTEM,
        build_pair=_build_strict,
    )
)


# ===========================================================================
# concept_first_v1
#
# Identical guidance and response shape; only the ORDER differs. Putting the
# article first makes it a stable prefix across every concept tested against that
# document, which is what lets a server reuse the prompt's KV cache between the
# N calls for one article. strict_v1 puts the article first for exactly that
# reason; this variant puts the concept first so the effect can be measured
# rather than assumed.
# ===========================================================================


def _build_concept_first(
    document: Document, concept: Concept, body: str, body_limit: int
) -> str:
    return (
        f"=== Concept ===\n{_concept_block(concept)}\n\n"
        f"=== Article ===\n"
        f"url: {document.url}\n"
        f"title: {document.title or ''}\n"
        f"body: {_trim(body, body_limit)}\n\n"
        f"{RESPONSE_SHAPE}\n"
    )


register(
    PromptTemplate(
        prompt_id="concept_first_v1",
        mode="per-pair",
        system=_STRICT_SYSTEM,
        build_pair=_build_concept_first,
    )
)


# ===========================================================================
# lenient_v1
#
# Deliberately omits the exclusion list, as a baseline. Its purpose is to make
# the exclusions' contribution measurable: if strict_v1 does not beat this on
# precision, the extra instructions are not earning their tokens.
# ===========================================================================

_LENIENT_SYSTEM = (
    "You decide whether a news article reports an occurrence of a concept. "
    "Output one JSON object and nothing else."
)

register(
    PromptTemplate(
        prompt_id="lenient_v1",
        mode="per-pair",
        system=_LENIENT_SYSTEM,
        build_pair=_build_strict,
    )
)


# ===========================================================================
# strict_batch_v1 — one call per document, judging every candidate at once.
#
# Same guidance as strict_v1, so the two are comparable and the only difference
# is construction. The trade is real in both directions: far fewer calls and one
# shared copy of the article body instead of N, against a longer single prompt
# and a response the model can get partially wrong.
#
# A model that omits a concept from its array would silently shrink the
# denominator, so every requested concept is filled in — see `parse_batch` in
# `judge.run`, which supplies a zero-confidence non-match for anything missing.
#
# Self-consistency sampling does not apply here: repeating a batch call re-rolls
# every verdict together, so the votes are not independent. Batch runs are a
# single greedy call regardless of `judge.samples`.
# ===========================================================================

BATCH_RESPONSE_SHAPE = (
    "Output format, a single JSON object:\n"
    '{"verdicts": [\n'
    '  {"concept_id": <int>, "matched": <true|false>, "confidence": <0..1>, '
    '"country": "<two-letter ISO country code, or empty>", '
    '"evidence": "<the exact passage from the article, or empty>"},\n'
    "  ... one entry for EVERY concept listed above ...\n"
    "]}"
)


def _build_batch_strict(
    document: Document, concepts: list[Concept], body: str, body_limit: int
) -> str:
    blocks = "\n\n".join(
        f"[concept_id {concept.id}]\n{_concept_block(concept)}" for concept in concepts
    )
    return (
        f"=== Article ===\n"
        f"url: {document.url}\n"
        f"title: {document.title or ''}\n"
        f"body: {_trim(body, body_limit)}\n\n"
        f"You will judge EACH of the following {len(concepts)} concepts "
        f"independently against the article above. A concept matching does not "
        f"make another more or less likely.\n\n"
        f"=== Concepts ===\n{blocks}\n\n"
        f"{BATCH_RESPONSE_SHAPE}\n"
    )


register(
    PromptTemplate(
        prompt_id="strict_batch_v1",
        mode=PER_DOCUMENT,
        system=_STRICT_SYSTEM,
        build_pair=_build_strict,
        build_batch=_build_batch_strict,
    )
)


# The hierarchical arm is asked exactly as the batched baseline is: the same
# system prompt, the same builder, the same response shape. Only which classes
# go into each call differs, so a difference between the arms is the hierarchy's
# and not the wording's.
register(
    PromptTemplate(
        prompt_id="hier_batch_v1",
        mode=HIERARCHICAL,
        system=_STRICT_SYSTEM,
        build_pair=_build_strict,
        build_batch=_build_batch_strict,
    )
)


# ===========================================================================
# hier_batch_v2
#
# A parent class is a routing step, not a finding: it decides whether the
# classes below it are asked at all. A wrong yes costs one more call; a wrong
# no loses every leaf beneath it. The strict prompt leans the other way (it
# rejects warnings, threats and articles merely "about" a concept), which is
# right for leaves and wrong for routing. So parents are asked as questions,
# leaning to yes, in calls of their own; leaves are asked exactly as in
# hier_batch_v1 and the flat baseline, in separate calls, so leaf judging is
# unchanged and any difference between the arms comes from the hierarchy.
# ===========================================================================

_ROUTE_SYSTEM = (
    "You route a news article through a classification of supply-chain events. "
    "For each question, decide whether the article concerns it at all.\n"
    "Answer matched=true if any part of the article reports, describes, warns of "
    "or threatens what the question asks about, even briefly or in passing, and "
    "whether it has happened, is under way or is only expected. More specific "
    "classes are judged separately and strictly after this step, so when in "
    "doubt, answer true.\n"
    "Answer matched=false only when the article has nothing to do with the "
    "question, or the relevant words appear only in menus, boilerplate or a "
    "paywall notice.\n"
    "Copy the evidence exactly from the article text; do not summarise it or "
    "make it up. Give confidence as how likely your answer is to be right, "
    "from 0 to 1.\n"
    "Output one JSON object and nothing else."
)


def _build_route(
    document: Document, concepts: list[Concept], body: str, body_limit: int
) -> str:
    # A parent's text is its routing question, held as ontology content.
    blocks = "\n\n".join(
        f"[concept_id {concept.id}]\nquestion: {concept.definition or concept.name}"
        for concept in concepts
    )
    return (
        f"=== Article ===\n"
        f"url: {document.url}\n"
        f"title: {document.title or ''}\n"
        f"body: {_trim(body, body_limit)}\n\n"
        f"Answer EACH of the following {len(concepts)} questions independently "
        f"against the article above.\n\n"
        f"=== Questions ===\n{blocks}\n\n"
        f"{BATCH_RESPONSE_SHAPE.replace('concept listed', 'question listed')}\n"
    )


register(
    PromptTemplate(
        prompt_id="hier_batch_v2",
        mode=HIERARCHICAL,
        system=_STRICT_SYSTEM,
        build_pair=_build_strict,
        build_batch=_build_batch_strict,
        route_system=_ROUTE_SYSTEM,
        build_route=_build_route,
    )
)
