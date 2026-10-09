"""The arms report as Markdown tables, for the results write-up."""

from __future__ import annotations

from hontology.evaluation.labels.sample import EQUAL_PER_WINDOW


def _ci(value: float | None, ci: list[float | None] | tuple) -> str:
    if value is None:
        return "-"
    low, high = (list(ci) + [None, None])[:2]
    if low is None or high is None:
        return f"{value:.2f}"
    return f"{value:.2f} ({low:.2f}\u2013{high:.2f})"


def _rate_cell(stat: dict | None) -> str:
    if not stat or stat.get("n") in (None, 0):
        return "-"
    return f"{stat['hits']}/{stat['n']} = " + _ci(stat["rate"], stat.get("ci", [None, None]))


def render_markdown(report: dict) -> str:
    """The arms report as Markdown tables, for the results write-up.

    Every number carries its interval, and cost is shown beside the scores it
    bought. Article-level cost is on the labelled documents only, so an arm
    that judged more of the calendar does not look more expensive for it.
    """
    baseline = report["baseline"]
    runs = report["runs"]
    name = {
        run_id: f"{run_id} (baseline)" if run_id == baseline else str(run_id) for run_id in runs
    }
    out: list[str] = []
    out.append("| Run | Ontology version | Prompt |\n| --- | --- | --- |")
    for r in runs:
        version = runs[r].get("ontology_version") or "-"
        out.append(f"| {name[r]} | {version} | {runs[r].get('prompt_id') or '-'} |")

    sample = report.get("sample")
    if sample and any("article" in runs[r] for r in runs):
        out += _article_section(sample, runs, name)
    comparisons = report.get("comparisons") or {}
    if comparisons:
        out += _comparison_section(comparisons)
    out += _calendar_section(runs, name)
    return "\n".join(out) + "\n"


def _article_section(sample: dict, runs: dict, name: dict) -> list[str]:
    """End-to-end and judge-only scores on the labelled sample, and the stopping rule."""
    out: list[str] = []
    status = sample.get("status") or {}
    out.append(
        f"\n## Article level\n\nLabelled sample: the first {sample['labelled_prefix']} "
        "documents of the frozen order "
        f"(manifest {str(sample.get('manifest_sha256'))[:12]}). "
        "Intervals are 95%, resampling whole documents. End to end, a pair never "
        "judged counts as no; judge only scores the pairs each run judged."
        + (
            " Every calendar window has an equal share of the sample, so each "
            "document is weighted by its window's size over the number labelled "
            "from it, and the bootstrap resamples within windows; the McNemar "
            "counts below are unweighted."
            if sample.get("allocation") == EQUAL_PER_WINDOW
            else ""
        )
    )
    out.append(
        "\n| Run | Precision | Recall | F1 | Judge-only precision | Judge-only recall "
        "| Pairs judged | Tokens on sample | Seconds on sample |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in runs:
        a = runs[r].get("article")
        if not a:
            continue
        e, j, c = a["end_to_end"], a["judge_only"], a.get("cost_on_sample") or {}
        tokens = (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
        out.append(
            f"| {name[r]} | {_ci(e['precision'], e['precision_ci'])} "
            f"| {_ci(e['recall'], e['recall_ci'])} | {_ci(e['f1'], e['f1_ci'])} "
            f"| {_ci(j['precision'], j['precision_ci'])} "
            f"| {_ci(j['recall'], j['recall_ci'])} "
            f"| {a['pairs_judged']} | {tokens:,} | {c.get('seconds', 0):,.0f} |"
        )
    if status:
        widths = status.get("half_widths") or {}
        out.append(
            f"\nStopping rule: the baseline's 95% intervals must reach "
            f"\u00b1{status.get('target', 0.05):.2f}. "
            f"Now \u00b1{widths.get('precision') or 0:.2f} "
            f"on precision and \u00b1{widths.get('recall') or 0:.2f} on recall, over "
            f"{status.get('positives')} positive pairs: "
            f"{'met' if status.get('target_met') else 'not met, keep labelling'}."
        )
    return out


def _comparison_section(comparisons: dict) -> list[str]:
    """Each arm against the baseline, then any hierarchy diagnostics."""
    out: list[str] = []
    out.append(
        "\n## Against the baseline\n\nAn arm counts as an improvement only if the "
        "paired interval on its F1 difference lies above zero.\n\n"
        "| Arm | F1 difference | Improvement | Discordant pairs | Only baseline right "
        "| Only arm right | McNemar p |\n| --- | --- | --- | --- | --- | --- | --- |"
    )
    for r, comparison in comparisons.items():
        f1, mc = comparison["f1"], comparison["mcnemar"]
        p = "-" if mc.get("p_value") is None else f"{mc['p_value']:.3f}"
        out.append(
            f"| {r} | {_ci(f1['difference'], f1['difference_ci'])} "
            f"| {'yes' if f1['improvement'] else 'no'} | {mc['discordant']} "
            f"| {mc['only_a_correct']} | {mc['only_b_correct']} | {p} |"
        )
    for r, comparison in comparisons.items():
        h = comparison.get("hierarchy")
        if not h:
            continue
        out.append(
            f"\n### Hierarchy diagnostics, run {r}\n\n"
            "Recall at each level counts only classes whose parent was answered yes, "
            "so a miss shows at the level where it happened.\n\n"
            "| Level | Recall | Hits / positives |\n| --- | --- | --- |"
        )
        for level, row in h["recall_by_level"].items():
            value = "-" if row.get("recall") is None else f"{row['recall']:.2f}"
            out.append(f"| {level} | {value} | {row['hits']}/{row['n']} |")
        cov = h["coverage"]
        out.append(
            f"\nParent classes answered yes: {h['internal_answered_yes']}, of which "
            f"{h['internal_false_positives']} with no labelled leaf below "
            "(gap candidates). "
            f"Correct leaf matches: {cov['true_positives']}, of which "
            f"{cov['beyond_baseline_retrieval']} on pairs the baseline's retrieval had "
            "not selected (coverage, not structure)."
        )
    return out


def _calendar_section(runs: dict, name: dict) -> list[str]:
    """Event-level recall and false alarms, raw and verified, with judging cost."""
    out = [
        "\n## Calendar\n\nVerified counts a detection only once a person has confirmed it "
        "is the calendar's event; until matches are reviewed, verified rates stay at "
        "zero, so the raw rate is shown beside each. Rates carry Wilson 95% "
        "intervals.\n\n"
        "| Run | | Events found | Precursors found | False alarms on controls "
        "| Verdicts | Tokens | Seconds |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    for r in runs:
        cal, cost = runs[r]["calendar"], runs[r]["cost"]
        v = cal.get("verified") or {}
        tokens = cost["input_tokens"] + cost["output_tokens"]
        out.append(
            f"| {name[r]} | raw | {_rate_cell(cal.get('event_recall'))} "
            f"| {_rate_cell(cal.get('precursor_recall'))} "
            f"| {_rate_cell(cal.get('false_alarm_rate'))} "
            f"| {cost['pairs']:,} | {tokens:,} | {cost['seconds']:,.0f} |"
        )
        out.append(
            f"| | verified | {_rate_cell(v.get('event_recall'))} "
            f"| {_rate_cell(v.get('precursor_recall'))} "
            f"| {_rate_cell(v.get('false_alarm_rate'))} | | | |"
        )
    return out
