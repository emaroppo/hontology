"""Routing an article down the class hierarchy: arm H's retrieval.

The top-level classes are asked first; then, round by round, the not-yet-asked
children of every parent answered yes, one set per parent, until nothing new
opens. In each set the parents are asked as routing questions, in one call;
the leaves are not asked here at all: they become an offer, the leaves one
deciding call is shown.

Only parents open further sets, so every routing call can be made before any
leaf is decided. The sets are the ones the interleaved descent of earlier runs
asked, so the deciding calls are too.

- **Each class at most once.** A class with two parents answered yes is placed
  under the first only: a second parent means *either*, so one answer serves both.
- **No descent below a failure.** A failed routing call counts as not positive,
  so nothing beneath it is offered rather than guessed at.
- **Resumable.** Parent answers a run already stored are passed in as *known*
  and not asked again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hontology.pipeline.retrieve.strategies.base import Model, Offer, Retrieval


@dataclass
class Route:
    """Routing step: offers the leaves the parents answered yes lead to."""

    model: Model
    top: list[int]
    children: dict[int, set[int]]
    # Parent answers already stored, from an interrupted run.
    known: dict[int, bool | None] = field(default_factory=dict)
    name: str = "routing"

    def __call__(self, retrieval: Retrieval) -> None:
        from hontology.pipeline.judge.batched import ask_batch

        document, body = retrieval.article.document, retrieval.article.body
        assert body is not None
        # Parents' answers, and every leaf already placed in an offer (False:
        # a leaf opens nothing).
        placed: dict[int, bool | None] = {
            c: yes for c, yes in self.known.items() if self.children.get(c)
        }
        while sets := next_sets(self.top, self.children, placed):
            for class_ids in sets:
                parents = [c for c in class_ids if self.children.get(c)]
                leaves = [c for c in class_ids if not self.children.get(c)]
                if parents:
                    asked = ask_batch(
                        self.model, document, self.model.concepts(parents), body, routing=True
                    )
                    if asked.completion is not None:
                        retrieval.spend.add(asked.completion)
                    retrieval.asked.append(asked)
                    placed.update({c: asked.answer(c) for c in parents})
                if leaves:
                    retrieval.offers.append(Offer(leaves))
                    placed.update(dict.fromkeys(leaves, False))
        retrieval.note(
            self.name,
            calls=len(retrieval.asked),
            failed=sum(a.error is not None for a in retrieval.asked),
            yes=sorted(c for a in retrieval.asked for c in a.concept_ids if a.answer(c)),
            offers=len(retrieval.offers),
        )


def next_sets(
    top: list[int], children: dict[int, set[int]], answered: dict[int, bool | None]
) -> list[list[int]]:
    """The sibling sets to ask next: unasked top classes, and the unasked
    children of every class answered yes, each class in one set at most."""
    sets: list[list[int]] = []
    queued: set[int] = set()
    pending_top = [c for c in top if c not in answered]
    if pending_top:
        sets.append(pending_top)
        queued.update(pending_top)
    for parent in sorted(c for c, yes in answered.items() if yes):
        below = sorted(
            c for c in children.get(parent, ()) if c not in answered and c not in queued
        )
        if below:
            sets.append(below)
            queued.update(below)
    return sets
