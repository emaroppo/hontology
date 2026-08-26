"""Ontology health checks.

Two failure modes that cost labelling hours to discover downstream, and seconds
to catch here.

**Strength drift.** A concept named "*Successful* negotiation" whose definition
only requires "a concrete commitment" will match a mere promise — and the model
is *right* to do so, because the prompt tells it to follow the definition over
the name. The bug is in the ontology, not the judge, but it surfaces as
inexplicable false positives. The check looks for strength words in a name that
the definition never earns.

**Near-duplicates.** Two concepts whose definitions embed almost identically
cannot be told apart by retrieval, so labels for one contaminate the other and
per-concept metrics become meaningless. Cosine similarity over the stored
concept vectors finds them.

Both are warnings, never errors. An ontology is the user's to author, and a lint
that blocks is a lint that gets ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Concept, Embedding, EmbeddingModel

# Words in a NAME that promise a stronger claim than a vague definition delivers.
STRENGTH_CUES = (
    "successful",
    "successfully",
    "major",
    "mass",
    "severe",
    "significant",
    "complete",
    "total",
    "full",
    "violent",
    "large-scale",
    "widespread",
    "unprecedented",
    "critical",
    "collapse",
    "halt",
    "shutdown",
)

# Words a definition can use to earn a strong name.
COMMITMENT_CUES = (
    "must",
    "at least",
    "more than",
    "all ",
    "entire",
    "complet",
    "halt",
    "ceas",
    "stop",
    "suspend",
    "kill",
    "destroy",
    "actually",
    "confirm",
    "verif",
    "resulted in",
    "no longer",
    "large",
    "severe",
)

DEFAULT_DUPLICATE_THRESHOLD = 0.92
SHORT_DEFINITION_CHARS = 80

# Cues are matched on a stem, so a name saying "violent" is satisfied by a
# definition saying "violence", and "shutdown" by "halts operations". Without
# this the check fires on definitions that plainly do earn their name, and a
# lint that cries wolf is a lint nobody reads.
_STEM_LENGTH = 5


def _mentions(body: str, cue: str) -> bool:
    """Whether *body* contains *cue* or an obvious morphological variant."""
    cue = cue.lower().strip()
    if cue in body:
        return True
    stem = cue[:_STEM_LENGTH]
    return len(stem) >= _STEM_LENGTH and stem in body


@dataclass
class Finding:
    check: str
    severity: str
    concept_id: int
    concept_name: str
    message: str
    related_concept_id: int | None = None
    score: float | None = None

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "message": self.message,
            "related_concept_id": self.related_concept_id,
            "score": self.score,
        }


def check_missing_text(concepts: list[Concept]) -> list[Finding]:
    """A concept with no definition is judged on its name alone."""
    findings: list[Finding] = []
    for concept in concepts:
        definition = (concept.definition or "").strip()
        if not definition:
            findings.append(
                Finding(
                    check="missing_definition",
                    severity="warning",
                    concept_id=concept.id,
                    concept_name=concept.name,
                    message=(
                        "no definition — the judge will have only the name to go on, "
                        "and retrieval will embed the name alone"
                    ),
                )
            )
        elif len(definition) < SHORT_DEFINITION_CHARS:
            findings.append(
                Finding(
                    check="thin_definition",
                    severity="info",
                    concept_id=concept.id,
                    concept_name=concept.name,
                    message=f"definition is only {len(definition)} characters; "
                    "short definitions retrieve broadly and judge inconsistently",
                )
            )
        if not (concept.exclusion_criteria or "").strip():
            findings.append(
                Finding(
                    check="no_exclusions",
                    severity="info",
                    concept_id=concept.id,
                    concept_name=concept.name,
                    message=(
                        "no exclusion criteria — usually the highest-leverage edit "
                        "available for precision"
                    ),
                )
            )
    return findings


def check_strength_drift(concepts: list[Concept]) -> list[Finding]:
    """Names promising more than their definitions require."""
    findings: list[Finding] = []
    for concept in concepts:
        name = (concept.name or "").lower()
        definition = (concept.definition or "").lower()
        criteria = (concept.inclusion_criteria or "").lower()
        if not definition:
            continue

        promised = [cue for cue in STRENGTH_CUES if cue in name]
        if not promised:
            continue

        body = f"{definition} {criteria}"
        # The definition earns the name if it echoes the strength word (in any
        # obvious variant) or otherwise commits to a threshold.
        earned = any(_mentions(body, cue) for cue in promised) or any(
            _mentions(body, cue) for cue in COMMITMENT_CUES
        )
        if not earned:
            findings.append(
                Finding(
                    check="strength_drift",
                    severity="warning",
                    concept_id=concept.id,
                    concept_name=concept.name,
                    message=(
                        f"the name promises {promised!r} but the definition sets no "
                        "matching threshold. The judge follows the definition over the "
                        "name, so weaker events will match and read as false positives"
                    ),
                )
            )
    return findings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def check_near_duplicates(
    session: Session,
    ontology_id: int,
    *,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    embed_model: str | None = None,
) -> list[Finding]:
    """Concepts whose stored vectors are too close to tell apart.

    Uses whatever vectors already exist rather than embedding on demand, so lint
    stays free. Returns nothing when the ontology has never been embedded.
    """
    concepts = {
        c.id: c
        for c in session.scalars(select(Concept).where(Concept.ontology_id == ontology_id))
    }
    if len(concepts) < 2:
        return []

    query = select(Embedding).where(
        Embedding.object_type == "concept", Embedding.object_id.in_(concepts)
    )
    if embed_model:
        model_row = session.scalar(
            select(EmbeddingModel).where(EmbeddingModel.model_name == embed_model)
        )
        if model_row is None:
            return []
        query = query.where(Embedding.model_id == model_row.id)

    # Keep one vector per concept — the most recent, which reflects current wording.
    vectors: dict[int, list[float]] = {}
    for row in session.scalars(query.order_by(Embedding.id)):
        vectors[row.object_id] = list(row.embedding)

    findings: list[Finding] = []
    ids = sorted(vectors)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            score = _cosine(vectors[left], vectors[right])
            if score < threshold:
                continue
            findings.append(
                Finding(
                    check="near_duplicate",
                    severity="warning",
                    concept_id=left,
                    concept_name=concepts[left].name,
                    related_concept_id=right,
                    score=round(score, 4),
                    message=(
                        f"embeds almost identically to {concepts[right].name!r} "
                        f"(cosine {score:.3f}). Retrieval cannot separate them, so "
                        "labels for one contaminate the other and per-concept "
                        "metrics stop meaning anything"
                    ),
                )
            )
    return findings


def lint(
    session: Session,
    ontology_id: int,
    *,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> dict:
    """Run every check. Findings are advisory, never blocking."""
    concepts = list(session.scalars(select(Concept).where(Concept.ontology_id == ontology_id)))
    findings = (
        check_missing_text(concepts)
        + check_strength_drift(concepts)
        + check_near_duplicates(session, ontology_id, threshold=duplicate_threshold)
    )
    order = {"warning": 0, "info": 1}
    findings.sort(key=lambda f: (order.get(f.severity, 2), f.concept_name))

    return {
        "ontology_id": ontology_id,
        "concepts": len(concepts),
        "findings": [f.as_dict() for f in findings],
        "warnings": sum(1 for f in findings if f.severity == "warning"),
        "info": sum(1 for f in findings if f.severity == "info"),
    }
