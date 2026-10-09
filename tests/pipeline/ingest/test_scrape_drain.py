"""Scraping batch after batch, without stopping early on held-back documents."""

from __future__ import annotations

from hontology.pipeline.ingest.articles.scrape import drain


def batch(attempted: int = 0, held_back: list[int] | None = None, deferred: int = 0) -> dict:
    return {"attempted": attempted, "deferred": deferred, "held_back": held_back or []}


class Script:
    """Hands out prepared batch results and records what each call held back."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.held: list[list[int]] = []

    def __call__(self, held: list[int]) -> dict:
        self.held.append(held)
        return self.results.pop(0) if self.results else batch()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def run(script: Script, clock: Clock, budget: int = 1000) -> dict:
    return drain(script, budget=budget, hold_back_s=600, clock=clock, sleep=clock.sleep)


def test_held_back_documents_stay_out_of_the_next_batches():
    script = Script([batch(attempted=5, held_back=[1, 2]), batch(attempted=3), batch()])
    run(script, Clock())
    assert script.held[1] == [1, 2]


def test_a_batch_of_only_deferred_documents_does_not_end_the_work():
    """The bug: a batch that attempted nothing because every document in it
    was deferred read as "nothing left", and the scrape stopped."""
    clock = Clock()
    script = Script([batch(held_back=[7], deferred=1), batch(attempted=1), batch()])
    totals = run(script, clock)
    assert clock.slept == [600]  # waited for the held document to come due
    assert totals["attempted"] == 1
    assert script.held[1] == []  # it came due and was offered again


def test_nothing_attempted_and_nothing_held_ends_the_work():
    clock = Clock()
    script = Script([batch(attempted=2), batch()])
    run(script, clock)
    assert len(script.held) == 2
    assert clock.slept == []


def test_the_budget_ends_the_work():
    script = Script([batch(attempted=150), batch(attempted=150), batch(attempted=150)])
    totals = run(script, Clock(), budget=300)
    assert totals["attempted"] == 300
