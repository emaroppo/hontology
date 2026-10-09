"""Confusion counts and the rates derived from them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else None

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.total if self.total else None

    def add(self, *, expected: bool, predicted: bool) -> None:
        if expected and predicted:
            self.tp += 1
        elif expected and not predicted:
            self.fn += 1
        elif not expected and predicted:
            self.fp += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }
