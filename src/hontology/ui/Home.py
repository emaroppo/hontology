"""Landing page: the scored runs, end to end.

The app starts with nothing in it, the premise being that the ontology is yours,
so with no ontology this page makes the empty state actionable. Otherwise it
opens on the leaderboard: every judged run scored on the same labelled sample,
the way the arms comparison scores it.

- **Leaderboard**: one row per run, with its stage versions, how much of the
  sample it covered, end-to-end precision, recall and F1, and its cost.
- **Comparison**: arms against a baseline, paired, as `eval arms` reports them.
- **Single run**: one run end to end, with a couple of numbers per stage; each
  stage's own evaluation lives on its page.
"""

from __future__ import annotations

import streamlit as st

from hontology.ui import shared
from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="hontology", page_icon="🧭", layout="wide")

api = Api()

st.title("🧭 hontology")
st.caption(
    "Define an ontology. Detect it in a live news feed. Find out honestly how well that worked."
)

if not api.healthy():
    st.error(
        f"The API is not reachable at `{api.base_url}`.\n\n"
        "Start it with `make api`, then reload this page."
    )
    st.stop()

ontologies = api.list_ontologies()


def how_it_fits() -> None:
    with st.expander("How the pieces fit together"):
        st.markdown(
            """
            1. **Ontology**: describe what you care about: classes, their
               definitions, and the criteria that draw their boundaries.
            2. **Filtering**: link classes to the codes the feed tags articles
               with, so only articles worth reading are downloaded.
            3. **Retrieval**: rank classes against each article, and cut.
            4. **Judgement**: ask a model whether each article reports each class.
            5. **Labelling**: build ground truth, a sample at a time.
            6. **Here**: score runs end to end; each stage page scores its own stage.
            """
        )


if not ontologies:
    st.info("**No ontologies yet.** This install is empty by design.")
    st.markdown("Head to the **Ontology** page to create one, or import one from OWL or JSON.")
    how_it_fits()
    st.stop()

params = shared.current(api)
ontology = next((o for o in ontologies if o["id"] == params.ontology_id), ontologies[0])
manifest_path = params.sample
annotator = params.annotator


def ci_text(value: float | None, ci: list | None) -> str:
    if value is None:
        return "—"
    if ci and ci[0] is not None:
        return f"{value:.2f} ({ci[0]:.2f}–{ci[1]:.2f})"
    return f"{value:.2f}"


try:
    board = api.live_leaderboard(ontology["id"], manifest_path, annotator)
except ApiError as exc:
    st.error(exc.detail)
    how_it_fits()
    st.stop()

runs = board["rows"]
calendar_cache: dict = st.session_state.setdefault("home_calendar", {})


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def show_leaderboard() -> None:
    labelled = board["labelled_articles"]
    if not labelled:
        st.info(
            "No article of this sample is labelled yet, so nothing can be scored. Label "
            "some on the **Labelling** page, or score against a machine annotation set "
            "(Truth, in Settings at the top right)."
        )
        return
    show_partial = st.toggle(
        "Include runs that did not cover the sample",
        value=False,
        help="A run made for another calendar processed few of these articles, so its "
        "recall here says nothing about it.",
    )
    shown = [r for r in runs if show_partial or r["covered"] == labelled]
    shown.sort(key=lambda r: -((r["end_to_end"] or {}).get("f1") or -1))
    if not shown:
        st.info("No run covered every labelled article. Include partial runs to see them.")
        return
    table = []
    for r in shown:
        e2e = r["end_to_end"] or {}
        cost = r["cost"] or {}
        calendar = calendar_cache.get(r["run_id"])
        recall = (calendar or {}).get("event_recall") or {}
        table.append(
            {
                "Run": f"{r['run_id']} · {r['name']}",
                "Prompt": r["versions"]["prompt_id"],
                "F1": ci_text(e2e.get("f1"), e2e.get("f1_ci")),
                "Precision": ci_text(e2e.get("precision"), e2e.get("precision_ci")),
                "Recall": ci_text(e2e.get("recall"), e2e.get("recall_ci")),
                "Covered": f"{r['covered']}/{labelled}",
                "Pairs judged": r["pairs_judged"],
                "Tokens": (cost.get("input_tokens") or 0) + (cost.get("output_tokens") or 0),
                "Seconds": round(cost.get("seconds") or 0),
                "Calendar recall": ci_text(recall.get("rate"), recall.get("ci"))
                if calendar
                else "",
                "Filter": ", ".join(r["versions"]["filter"]) or "not recorded",
                "Retrieval": r["versions"]["retrieval"],
                "Judge": r["versions"]["judge"],
            }
        )
    st.dataframe(table, hide_index=True)
    st.caption(
        f"End to end on the first {board['labelled_prefix']} articles of the sample, "
        f"{labelled} of them labelled: a leaf a run never judged counts as no, and each "
        "article is weighted by its calendar window, as in the arms report. Intervals are "
        "95%, resampling articles. Tokens and seconds are judging on these articles."
        + (
            f" {board['out_of_turn']} article(s) labelled out of order are left out."
            if board["out_of_turn"]
            else ""
        )
    )
    with_calendar = [r for r in shown if r["calendar"] and r["run_id"] not in calendar_cache]
    if with_calendar:
        cols = st.columns([3, 1])
        pick = cols[0].multiselect(
            "Add calendar recall (runs that worked through a calendar; tens of seconds each)",
            [r["run_id"] for r in with_calendar],
            format_func=lambda i: f"run {i}",
        )
        if pick and cols[1].button("Score on their calendars"):
            for run_id in pick:
                try:
                    with st.spinner(f"Run {run_id}: walking its calendar…"):
                        calendar_cache[run_id] = api.run_calendar(run_id)
                except ApiError as exc:
                    st.error(exc.detail)
            st.rerun()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def show_comparison() -> None:
    labels = {
        r["run_id"]: f"{r['run_id']} · {r['name']} ({r['versions']['prompt_id']})" for r in runs
    }
    if len(labels) < 2:
        st.info("Comparing needs at least two judged runs.")
        return
    ids = list(labels)
    cols = st.columns(2)
    baseline = cols[0].selectbox(
        "Baseline", ids, format_func=lambda i: labels[i], key="cmp_baseline"
    )
    if baseline is None:
        return
    arms = cols[1].multiselect(
        "Arms",
        [i for i in ids if i != baseline],
        format_func=lambda i: labels[i],
        key=f"cmp_arms_{baseline}",
    )
    st.caption(
        "Each arm against the baseline on the same labelled articles, paired: an arm "
        "counts as an improvement only if the interval on its F1 difference lies above "
        "zero. The same tables `hontology eval arms` writes."
    )
    if not arms:
        return
    try:
        result = api.compare_arms(baseline, arms, manifest_path, annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.markdown(result["markdown"])
    st.download_button(
        "Download as Markdown",
        data=result["markdown"],
        file_name=f"arms-{baseline}-vs-{'-'.join(map(str, arms))}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def show_single() -> None:
    labels = {r["run_id"]: f"{r['run_id']} · {r['name']} ({r['status']})" for r in runs}
    run_id = params.run_id
    if run_id is None or run_id not in labels:
        st.info("Pick a run that has judged something in Settings, at the top right.")
        return
    st.markdown(f"**{labels[run_id]}**")
    row = next(r for r in runs if r["run_id"] == run_id)

    st.markdown("**End to end**")
    e2e = row["end_to_end"] or {}
    cols = st.columns(4)
    cols[0].metric("F1", ci_text(e2e.get("f1"), e2e.get("f1_ci")))
    cols[1].metric("Precision", ci_text(e2e.get("precision"), e2e.get("precision_ci")))
    cols[2].metric("Recall", ci_text(e2e.get("recall"), e2e.get("recall_ci")))
    cols[3].metric("Sample covered", f"{row['covered']}/{board['labelled_articles']}")
    if row["calendar"]:
        calendar = calendar_cache.get(run_id)
        if calendar is None:
            if st.button(f"Score on its calendar ({row['calendar']}; tens of seconds)"):
                try:
                    with st.spinner("Walking the calendar…"):
                        calendar_cache[run_id] = api.run_calendar(run_id)
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)
        else:
            cols = st.columns(4)
            for col, name, key in (
                (cols[0], "Event recall", "event_recall"),
                (cols[1], "Precursor recall", "precursor_recall"),
                (cols[2], "False alarms on controls", "false_alarm_rate"),
            ):
                stat = calendar.get(key) or {}
                col.metric(name, ci_text(stat.get("rate"), stat.get("ci")))
            verified = (calendar.get("verified") or {}).get("event_recall") or {}
            cols[3].metric(
                "Verified event recall", ci_text(verified.get("rate"), verified.get("ci"))
            )
            if calendar.get("positives_lost_at"):
                st.caption(
                    "Events missed, by the stage that lost them: "
                    + ", ".join(f"{k} {v}" for k, v in calendar["positives_lost_at"].items())
                )

    st.markdown("**By stage**")
    versions = row["versions"]
    cols = st.columns(3)
    with cols[0], st.container(border=True):
        st.markdown("Filtering")
        st.caption("Links " + (", ".join(versions["filter"]) or "not recorded for this run"))
        try:
            flow = api.run_funnel(run_id)
            steps = {s["name"]: s["count"] for s in flow["documents"]}
            st.metric(
                "Articles this run considered", f"{steps.get('considered by this run', 0):,}"
            )
        except ApiError as exc:
            st.error(exc.detail)
        st.page_link("views/2_Filtering.py", label="Filtering evaluation", icon="🧹")
    with cols[1], st.container(border=True):
        st.markdown("Retrieval")
        source = versions["retrieval_source_run"]
        st.caption(
            f"Version {versions['retrieval']}"
            + (f", reused from run {source}" if source != run_id else "")
        )
        try:
            cut = api.retrieval_report(source, None, annotator)["run"]
            labelled = cut["labels"]
            st.metric(
                "True matches past the cutoff",
                f"{labelled['positives_kept']} of {labelled['positives']}"
                if labelled["positives"]
                else "—",
            )
            st.caption(f"{cut['pairs_per_document']:.2f} pairs per article to the judge")
        except ApiError as exc:
            st.error(exc.detail)
        st.page_link("views/3_Retrieval.py", label="Retrieval evaluation", icon="🔎")
    with cols[2], st.container(border=True):
        st.markdown("Judgement")
        st.caption(f"Version {versions['judge']} · {versions['prompt_id']}")
        judge = row["judge_only"] or {}
        st.metric(
            "Judge-only precision / recall",
            f"{ci_text(judge.get('precision'), None)} / {ci_text(judge.get('recall'), None)}",
            help="On the labelled pairs this run judged: what retrieval passed it.",
        )
        st.caption(f"{row['pairs_judged'] or 0} labelled pairs judged")
        st.page_link("views/4_Judgement.py", label="Judgement evaluation", icon="⚖️")

    with st.expander("Where it disagrees with the labels"):
        try:
            sample = api.run_sample(run_id, manifest_path, annotator)
            errors = sample.get("errors") or []
            st.dataframe(
                [
                    {
                        "Position": e["position"],
                        "Kind": e["kind"],
                        "Class": e["concept"],
                        "Article": e["document_title"] or e["document_url"],
                        "Link": e["document_url"],
                    }
                    for e in errors
                ],
                hide_index=True,
                column_config={"Link": st.column_config.LinkColumn(display_text="open")},
            )
        except ApiError as exc:
            st.error(exc.detail)
    with st.expander("Funnel: where the volume went"):
        try:
            flow = api.run_funnel(run_id)
            fcols = st.columns(2)
            for col, key in ((fcols[0], "documents"), (fcols[1], "pairs")):
                for step in flow[key]:
                    share = (
                        f"{step['of_previous']:.1%} of previous"
                        if step["of_previous"] is not None
                        else "start"
                    )
                    col.markdown(f"`{step['count']:>9,}`  {step['name']}: {share}")
        except ApiError as exc:
            st.error(exc.detail)
    with st.expander("Detections"):
        try:
            found = api.run_detections(run_id)
            stats = found["summary"]
            dcols = st.columns(3)
            dcols[0].metric("Detections", stats["detections"])
            dcols[1].metric("Events", stats["events"], help="Distinct (class, place, date).")
            dcols[2].metric("Confirmed", stats["by_verification"].get("confirmed", 0))
            if found["rows"]:
                st.download_button(
                    "Download detections (CSV)",
                    data=api.detections_csv(run_id),
                    file_name=f"detections-run{run_id}.csv",
                    mime="text/csv",
                )
        except ApiError as exc:
            st.error(exc.detail)


leaderboard_tab, comparison_tab, single_tab = st.tabs(
    ["Leaderboard", "Comparison", "Single run"], key="home_tab", on_change="rerun"
)
if leaderboard_tab.open:
    with leaderboard_tab:
        show_leaderboard()
if comparison_tab.open:
    with comparison_tab:
        show_comparison()
if single_tab.open:
    with single_tab:
        show_single()

st.divider()
how_it_fits()
