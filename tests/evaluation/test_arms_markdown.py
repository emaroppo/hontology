"""The arms report as Markdown: every number with its interval, cost beside it."""

from __future__ import annotations

from hontology.evaluation.comparison.arms_report import render_markdown


def scores(p: float, r: float, f: float) -> dict:
    return {
        "documents": 120,
        "tp": 40,
        "fp": 20,
        "fn": 10,
        "precision": p,
        "recall": r,
        "f1": f,
        "precision_ci": [p - 0.1, p + 0.1],
        "recall_ci": [r - 0.1, r + 0.1],
        "f1_ci": [f - 0.1, f + 0.1],
    }


def rate(hits: int, n: int) -> dict:
    return {"hits": hits, "n": n, "rate": hits / n, "ci": [0.2, 0.9]}


def run(version: str, prompt: str, *, p: float, sample_tokens: int) -> dict:
    calendar = {"event_recall": rate(10, 14), "precursor_recall": rate(4, 7)}
    calendar["false_alarm_rate"] = rate(2, 12)
    calendar["verified"] = dict(calendar)
    return {
        "ontology_version": version,
        "prompt_id": prompt,
        "calendar": calendar,
        "cost": {
            "pairs": 9000,
            "input_tokens": 1_000_000,
            "output_tokens": 200_000,
            "seconds": 3600,
        },
        "article": {
            "end_to_end": scores(p, 0.6, 0.6),
            "judge_only": scores(p + 0.05, 0.9, 0.7),
            "pairs_judged": 300,
            "cost_on_sample": {
                "pairs": 300,
                "input_tokens": sample_tokens,
                "output_tokens": 0,
                "seconds": 50.0,
            },
        },
    }


REPORT = {
    "baseline": 566,
    "arms": [600],
    "runs": {
        566: run("v1", "strict_batch_v1", p=0.58, sample_tokens=120_000),
        600: run("v3", "hier_batch_v2", p=0.71, sample_tokens=250_000),
    },
    "sample": {
        "manifest_sha256": "abcdef0123456789",
        "labelled_prefix": 120,
        "status": {
            "documents": 120,
            "positives": 50,
            "half_widths": {"precision": 0.1, "recall": 0.12},
            "target": 0.05,
            "target_met": False,
        },
    },
    "comparisons": {
        600: {
            "f1": {"difference": 0.08, "difference_ci": [0.01, 0.15], "improvement": True},
            "mcnemar": {
                "discordant": 30,
                "only_a_correct": 8,
                "only_b_correct": 22,
                "p_value": 0.017,
            },
            "hierarchy": {
                "recall_by_level": {0: {"n": 40, "hits": 38, "recall": 0.95}},
                "internal_answered_yes": 90,
                "internal_false_positives": 25,
                "coverage": {
                    "true_positives": 45,
                    "baseline_had_selected": 40,
                    "beyond_baseline_retrieval": 5,
                },
            },
        }
    },
}


def test_every_score_carries_its_interval():
    text = render_markdown(REPORT)
    assert "| 566 (baseline) | 0.58 (0.48–0.68)" in text
    assert "0.08 (0.01–0.15) | yes |" in text


def test_cost_is_shown_on_the_labelled_sample_beside_the_scores():
    text = render_markdown(REPORT)
    assert "| 300 | 250,000 | 50 |" in text  # the arm's tokens on the sample
    assert "| 300 | 120,000 | 50 |" in text


def test_provenance_and_stopping_rule_are_stated():
    text = render_markdown(REPORT)
    assert "| 600 | v3 | hier_batch_v2 |" in text
    assert "manifest abcdef012345" in text
    assert "not met, keep labelling" in text


def test_hierarchy_diagnostics_and_calendar_are_reported():
    text = render_markdown(REPORT)
    assert "25 with no labelled leaf below (gap candidates)" in text
    assert "5 on pairs the baseline's retrieval had not selected" in text
    assert "| 566 (baseline) | raw | 10/14 = 0.71 (0.20–0.90)" in text
    assert "| | verified | 10/14" in text


def test_a_report_without_labels_still_renders_the_calendar():
    bare = {"baseline": 566, "arms": [], "runs": {566: REPORT["runs"][566]}, "comparisons": {}}
    text = render_markdown(bare)
    assert "## Calendar" in text
    assert "## Article level" not in text
