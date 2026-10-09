"""What every judging mode needs: how to ask the judge, and how to record it."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from hontology.db.models import Concept, Document, Verdict
from hontology.judge import prompts
from hontology.judge.providers.base import Completion, GenerationConfig
from hontology.retrieve.candidates import document_body

NO_BODY = "no article body available"


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

    def spent(self, completion: Completion) -> None:
        self.input_tokens += completion.input_tokens or 0
        self.output_tokens += completion.output_tokens or 0


@dataclass
class Judging:
    """One run's judge: its template, model and settings, and its running tally."""

    session: Session
    run_id: int
    template: prompts.PromptTemplate
    provider: object
    judge_config: dict
    generation: GenerationConfig
    body_limit: int
    iso2_to_locus: dict[str, int]
    stats: JudgeStats

    def ask(self, system: str | None, prompt: str) -> Completion:
        return self.provider.complete(
            system=system,
            prompt=prompt,
            config=self.generation,
            want_json=True,
            want_reasoning=self.judge_config["think"],
            model=self.judge_config["model"],
        )

    def body(self, document: Document) -> str | None:
        return document_body(document, self.body_limit)

    def concepts(self, ids) -> list[Concept]:
        """The concepts with these ids, skipping any that no longer exist."""
        found = (self.session.get(Concept, concept_id) for concept_id in ids)
        return [concept for concept in found if concept is not None]

    def verdict(self, document_id: int, concept_id: int, *, samples: int = 1, **fields):
        return Verdict(
            run_id=self.run_id,
            document_id=document_id,
            concept_id=concept_id,
            provider=self.judge_config["provider"],
            model=self.judge_config["model"],
            prompt_id=self.template.prompt_id,
            mode=self.template.mode,
            samples=samples,
            **fields,
        )

    def failed(self, document_id: int, concept_ids, message: str) -> list[Verdict]:
        """Error rows, counted: the pairs stay in the denominator."""
        rows = [self.verdict(document_id, c, error=message) for c in concept_ids]
        self.stats.errors += len(rows)
        return rows


def share(total: int | None, parts: int, position: int) -> int:
    """One pair's share of a call's tokens; the first pair also takes the remainder.

    Splitting with plain integer division drops the remainder on every call, so
    a run's recorded tokens would fall short of what it actually spent.
    """
    whole, remainder = divmod(total or 0, parts)
    return whole + (remainder if position == 0 else 0)
