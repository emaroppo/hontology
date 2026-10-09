"""Checks on a concept's wording: missing text, and names that promise too much."""

from __future__ import annotations

from hontology.db.models import Concept
from hontology.ontology.lint.finding import Finding

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
