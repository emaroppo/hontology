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
    prompt_id: str
    mode: str  # "per-pair" | "per-document"
    system: str
    build_pair: Callable[[Document, Concept, str, int], str]


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
