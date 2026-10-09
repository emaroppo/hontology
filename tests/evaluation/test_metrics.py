"""Metrics and their intervals.

Hand-computed fixtures throughout: a metrics layer that is only tested against
itself can be confidently wrong.
"""

from __future__ import annotations

import pytest

from hontology.evaluation.metrics import (
    Confusion,
    bootstrap_f1,
    calibration_bins,
    mcnemar,
    retrieval_metrics,
    wilson,
    with_intervals,
)


class TestConfusion:
    def test_hand_computed_rates(self):
        c = Confusion(tp=8, fp=2, tn=6, fn=4)
        assert c.precision == pytest.approx(8 / 10)
        assert c.recall == pytest.approx(8 / 12)
        assert c.f1 == pytest.approx(2 * 8 / (2 * 8 + 2 + 4))
        assert c.accuracy == pytest.approx(14 / 20)

    def test_add_routes_each_case(self):
        c = Confusion()
        c.add(expected=True, predicted=True)
        c.add(expected=True, predicted=False)
        c.add(expected=False, predicted=True)
        c.add(expected=False, predicted=False)
        assert (c.tp, c.fn, c.fp, c.tn) == (1, 1, 1, 1)

    def test_empty_metrics_are_none_not_zero(self):
        """Zero would read as 'perfectly bad'; None reads as 'not measured'."""
        c = Confusion()
        assert c.precision is None
        assert c.recall is None
        assert c.f1 is None
        assert c.accuracy is None

    def test_no_predictions_leaves_precision_undefined(self):
        c = Confusion(tp=0, fp=0, tn=5, fn=3)
        assert c.precision is None
        assert c.recall == 0.0


class TestWilson:
    def test_interval_brackets_the_estimate(self):
        low, high = wilson(7, 10)
        assert low < 0.7 < high

    def test_stays_inside_zero_one_at_the_extremes(self):
        """Where the normal approximation produces impossible bounds."""
        low, high = wilson(0, 5)
        assert low == 0.0 and 0 < high < 1
        low, high = wilson(5, 5)
        assert 0 < low < 1 and high == 1.0

    def test_more_data_narrows_the_interval(self):
        narrow = wilson(70, 100)
        wide = wilson(7, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_empty_sample_is_undefined(self):
        assert wilson(0, 0) == (None, None)


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        c = Confusion(tp=20, fp=5, tn=20, fn=5)
        low, high = bootstrap_f1(c, n_boot=500, seed=1)
        assert low <= c.f1 <= high

    def test_is_reproducible_for_a_seed(self):
        c = Confusion(tp=20, fp=5, tn=20, fn=5)
        assert bootstrap_f1(c, n_boot=200, seed=7) == bootstrap_f1(c, n_boot=200, seed=7)

    def test_small_samples_give_wider_intervals(self):
        """The property that stops a 10-point gap on 30 pairs reading as a finding."""
        small = bootstrap_f1(Confusion(tp=4, fp=1, tn=4, fn=1), n_boot=500, seed=3)
        large = bootstrap_f1(Confusion(tp=40, fp=10, tn=40, fn=10), n_boot=500, seed=3)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_undefined_when_nothing_is_labelled(self):
        assert bootstrap_f1(Confusion(), n_boot=100) == (None, None)


class TestWithIntervals:
    def test_every_headline_metric_gets_an_interval(self):
        result = with_intervals(Confusion(tp=8, fp=2, tn=6, fn=4), seed=0)
        for key in ("precision_ci", "recall_ci", "f1_ci"):
            low, high = result[key]
            assert low is not None and high is not None and low <= high


class TestMcNemar:
    def test_counts_only_discordant_pairs(self):
        truth = {(1, 1): True, (1, 2): True, (1, 3): False, (1, 4): False}
        a = {(1, 1): True, (1, 2): True, (1, 3): False, (1, 4): True}
        b = {(1, 1): True, (1, 2): False, (1, 3): False, (1, 4): False}

        result = mcnemar(truth, a, b)
        assert result.n_pairs == 4
        assert result.both_correct == 2  # pairs 1 and 3
        assert result.only_a_correct == 1  # pair 2
        assert result.only_b_correct == 1  # pair 4
        # Concordant pairs carry no information about which run is better, which
        # is exactly what an unpaired test throws away.
        assert result.as_dict()["discordant"] == 2

    def test_identical_runs_have_no_signal(self):
        truth = {(1, i): i % 2 == 0 for i in range(10)}
        a = dict(truth)
        result = mcnemar(truth, a, dict(a))
        assert result.only_a_correct == 0
        assert result.only_b_correct == 0
        assert result.statistic is None
        assert result.p_value is None

    def test_a_lopsided_difference_is_significant(self):
        truth = {(1, i): True for i in range(40)}
        a = dict.fromkeys(truth, True)  # all correct
        b = {key: (i >= 20) for i, key in enumerate(truth)}  # 20 wrong
        result = mcnemar(truth, a, b)
        assert result.only_a_correct == 20
        assert result.only_b_correct == 0
        assert result.p_value is not None and result.p_value < 0.01

    def test_a_small_difference_is_not_significant(self):
        """The case that stops noise being reported as an improvement."""
        truth = {(1, i): True for i in range(30)}
        a = {key: i != 0 for i, key in enumerate(truth)}
        b = {key: i > 1 for i, key in enumerate(truth)}
        result = mcnemar(truth, a, b)
        assert result.p_value is None or result.p_value > 0.05

    def test_only_pairs_both_runs_judged_are_compared(self):
        truth = {(1, 1): True, (1, 2): True}
        a = {(1, 1): True, (1, 2): True}
        b = {(1, 1): False}
        assert mcnemar(truth, a, b).n_pairs == 1

    def test_unlabelled_pairs_are_ignored(self):
        truth = {(1, 1): True}
        a = {(1, 1): True, (1, 99): True}
        b = {(1, 1): True, (1, 99): False}
        assert mcnemar(truth, a, b).n_pairs == 1


class TestRetrieval:
    def test_recall_at_k_and_mrr(self):
        labels = {("d", 1): True, ("d", 2): True, ("d", 3): False}
        pool = {("d", 1): 1, ("d", 2): 4, ("d", 3): 2}
        metrics = retrieval_metrics(labels, pool, selected={("d", 1)}, ks=(1, 3, 5))

        assert metrics.recall_at_k[1] == pytest.approx(0.5)  # only pair 1 at rank 1
        assert metrics.recall_at_k[3] == pytest.approx(0.5)
        assert metrics.recall_at_k[5] == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx((1 / 1 + 1 / 4) / 2)

    def test_unlabelled_candidates_are_unknown_not_wrong(self):
        """The distinction that keeps precision readable on a sparse bank."""
        labels = {("d", 1): True}
        pool = {("d", 1): 1, ("d", 2): 2, ("d", 3): 3}
        metrics = retrieval_metrics(labels, pool, selected=set())

        assert metrics.precision == 1.0  # not 1/3
        assert metrics.coverage == pytest.approx(1 / 3)
        assert metrics.labelled_candidates == 1

    def test_cutoff_recall_is_separate_from_ranking(self):
        """Ranked well but cut anyway is a different failure from never ranked."""
        labels = {("d", 1): True, ("d", 2): True}
        pool = {("d", 1): 1, ("d", 2): 2}
        metrics = retrieval_metrics(labels, pool, selected={("d", 1)})

        assert metrics.recall_at_k[3] == 1.0  # ranking found both
        assert metrics.cutoff_recall == pytest.approx(0.5)  # the cutoff kept one

    def test_a_positive_never_ranked_lowers_recall(self):
        labels = {("d", 1): True, ("d", 2): True}
        pool = {("d", 1): 1}
        metrics = retrieval_metrics(labels, pool, selected={("d", 1)})
        assert metrics.n_positives == 2
        assert metrics.n_positives_ranked == 1
        assert metrics.recall_at_k[10] == pytest.approx(0.5)

    def test_empty_inputs_do_not_divide_by_zero(self):
        metrics = retrieval_metrics({}, {}, set())
        assert metrics.precision is None
        assert metrics.coverage is None
        assert metrics.mrr is None


class TestCalibration:
    def test_bins_report_observed_accuracy(self):
        pairs = [(0.9, True), (0.95, True), (0.9, False), (0.1, False), (0.1, False)]
        bins = calibration_bins(pairs, n_bins=5)

        top = bins[-1]
        assert top["n"] == 3
        assert top["accuracy"] == pytest.approx(2 / 3)

        bottom = bins[0]
        assert bottom["n"] == 2
        assert bottom["accuracy"] == 0.0

    def test_confidence_of_exactly_one_lands_in_the_top_bin(self):
        bins = calibration_bins([(1.0, True)], n_bins=4)
        assert bins[-1]["n"] == 1

    def test_empty_bins_report_none(self):
        bins = calibration_bins([(0.9, True)], n_bins=5)
        assert bins[0]["accuracy"] is None
