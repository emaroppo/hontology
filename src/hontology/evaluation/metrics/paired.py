"""Paired comparison of two runs: McNemar over the items both judged."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class McNemarResult:
    """Paired comparison over items both runs judged.

    ``b`` counts items the first run got right and the second wrong; ``c`` the
    reverse. Items both got right or both got wrong carry no information about
    which is better, which is precisely why an unpaired test wastes them.
    """

    both_correct: int = 0
    only_a_correct: int = 0  # b
    only_b_correct: int = 0  # c
    both_wrong: int = 0
    n_pairs: int = 0

    @property
    def discordant_count(self) -> int:
        """Pairs the two runs answered differently — the only informative ones."""
        return self.only_a_correct + self.only_b_correct

    @property
    def statistic(self) -> float | None:
        b, c = self.only_a_correct, self.only_b_correct
        if b + c == 0:
            return None
        # Continuity-corrected, appropriate for the small discordant counts a
        # hand-built label bank produces.
        return (abs(b - c) - 1) ** 2 / (b + c)

    @property
    def p_value(self) -> float | None:
        statistic = self.statistic
        if statistic is None:
            return None
        # Survival function of chi-square with one degree of freedom.
        return math.erfc(math.sqrt(max(0.0, statistic) / 2))

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "both_correct": self.both_correct,
            "only_a_correct": self.only_a_correct,
            "only_b_correct": self.only_b_correct,
            "both_wrong": self.both_wrong,
            "discordant": self.discordant_count,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }


def mcnemar(
    truth: dict[tuple[int, int], bool],
    a: dict[tuple[int, int], bool],
    b: dict[tuple[int, int], bool],
) -> McNemarResult:
    """Compare two runs on the pairs they both judged and that carry a label."""
    result = McNemarResult()
    for key, expected in truth.items():
        if key not in a or key not in b:
            continue
        result.n_pairs += 1
        a_right = a[key] == expected
        b_right = b[key] == expected
        if a_right and b_right:
            result.both_correct += 1
        elif a_right and not b_right:
            result.only_a_correct += 1
        elif b_right and not a_right:
            result.only_b_correct += 1
        else:
            result.both_wrong += 1
    return result
