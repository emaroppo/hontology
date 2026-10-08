"""Judgement: what the model says about a class, and how a prompt edit changes it.

Pick a class and a few articles, and the model is asked about each exactly as a
chosen run asks it: the same model, decoding settings and article text limit.
The prompt is shown in full and can be edited, both its system text and the
class's wording, and every article is then asked again with the edits beside the
original, against its label and what the run recorded.

Nothing a trial says is stored. Wording that works can be saved to the class,
which is then a new ontology version like any other edit.

Only per-pair prompts can be tried: one system text and one message per article
and class. Batched, hierarchical and extraction prompts are several calls whose
shape depends on earlier answers, so a run that used one is shown for what it
recorded, and the trial asks the per-pair way.
"""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from hontology.ui import charts, truth
from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Judgement", page_icon="⚖️", layout="wide")

api = Api()

st.title("⚖️ Judgement")

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

WORDING = {
    "definition": "Definition",
    "inclusion_criteria": "Counts when",
    "exclusion_criteria": "Does not count when",
}


# ---------------------------------------------------------------------------
# Sidebar: ontology, run (the model and settings), prompt, truth
# ---------------------------------------------------------------------------

with st.sidebar:
    by_label = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    ontology = by_label[st.selectbox("Ontology", list(by_label), key="judgement_ontology")]
    runs = api.judgement_runs(ontology["id"])
    if not runs:
        st.info("No run of this ontology has judged anything yet.")
        st.stop()
    by_run = {f"{r['id']} · {r['name']} ({r['prompt_id']})": r for r in runs}
    # A per-pair run compares like with like, so it is the default when there is one.
    templates = {t["prompt_id"]: t for t in api.judgement_templates()}
    default_run = next((i for i, r in enumerate(runs) if r["prompt_id"] in templates), 0)
    run = by_run[
        st.selectbox(
            "Ask as run",
            list(by_run),
            index=default_run,
            key="judgement_run",
            help="The model, decoding settings and article text limit come from this "
            "run, and its recorded verdicts are shown beside each trial.",
        )
    ]
    st.caption(f"{run['provider']} · {run['model']}")
    prompt_ids = list(templates)
    prompt_id = st.selectbox(
        "Prompt",
        prompt_ids,
        index=prompt_ids.index(run["prompt_id"]) if run["prompt_id"] in templates else 0,
        key="judgement_prompt",
    )
    if run["prompt_id"] not in templates:
        st.caption(
            f"Run {run['id']} used {run['prompt_id']}, which is not a per-pair prompt; "
            "its verdicts are shown for reference, and the trial asks per pair."
        )

    st.divider()
    annotator = truth.pick(api, ontology["id"], st, key="judgement_truth")


def show_trials() -> None:
    # ---------------------------------------------------------------------------
    # Class and prompt
    # ---------------------------------------------------------------------------

    leaves = api.leaves(ontology["id"])
    if not leaves:
        st.info("This ontology has no classes yet.")
        return
    by_name = {
        f"{leaf['families'][0]} › {leaf['name']}"
        if leaf["families"][0] != leaf["name"]
        else leaf["name"]: leaf
        for leaf in leaves
    }
    leaf = by_name[st.selectbox("Class", list(by_name), key="judgement_class")]

    # Edits start from the template and the class as saved; a new class, prompt or
    # run starts over.
    context = (leaf["id"], prompt_id)
    if st.session_state.get("judgement_context") != context:
        st.session_state["judgement_context"] = context
        st.session_state["edit_system"] = templates[prompt_id]["system"]
        for field in WORDING:
            st.session_state[f"edit_{field}"] = leaf.get(field) or ""

    prompt_col, articles_col = st.columns([1, 1])

    with prompt_col:
        st.markdown("**Prompt**")
        st.text_area("System text", key="edit_system", height=300)
        for field, label in WORDING.items():
            st.text_area(label, key=f"edit_{field}", height=90)
        if st.button("Undo my edits"):
            st.session_state.pop("judgement_context", None)
            st.rerun()

    system = st.session_state["edit_system"]
    wording = {field: st.session_state[f"edit_{field}"].strip() or None for field in WORDING}
    original_wording = {field: (leaf.get(field) or "").strip() or None for field in WORDING}
    system_edited = system != templates[prompt_id]["system"]
    wording_edited = wording != original_wording
    edited = system_edited or wording_edited

    def trial(document_id: int, *, mine: bool) -> dict:
        return {
            "run_id": run["id"],
            "document_id": document_id,
            "concept_id": leaf["id"],
            "prompt_id": prompt_id,
            "system": system if mine else None,
            "wording": wording if mine else None,
        }

    def cache_key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    answers: dict[str, dict] = st.session_state.setdefault("judgement_answers", {})

    # ---------------------------------------------------------------------------
    # Articles
    # ---------------------------------------------------------------------------

    with articles_col:
        st.markdown("**Articles**")
        try:
            articles = api.judgement_articles(run["id"], leaf["id"], annotator)
        except ApiError as exc:
            st.error(exc.detail)
            return
        if not articles:
            st.info(
                f"Run {run['id']} judged no article for this class, and none is labelled "
                "for it. Pick another class or run."
            )
            return

        def describe(article: dict) -> str:
            label = {True: "label ✓", False: "label ✗", None: "unlabelled"}[article["label"]]
            verdict = article["verdict"]
            said = (
                ""
                if verdict is None
                else " · run "
                + (
                    "✓"
                    if verdict["matched"]
                    else "✗"
                    if verdict["matched"] is False
                    else "error"
                )
            )
            return f"{(article['title'] or article['url'])[:80]} · {label}{said}"

        by_article = {describe(a): a for a in articles}
        chosen = st.multiselect(
            "Ask about",
            list(by_article),
            default=list(by_article)[:4],
            key=f"judgement_articles_{leaf['id']}_{run['id']}",
            help="Ordered most informative first: labelled true matches, then articles "
            "the run said match, then labelled articles the run judged.",
        )
        st.caption(
            "Each article is asked with the original prompt"
            + (" and with your edits" if edited else "")
            + ". Answers are kept for this session, so asking again costs nothing."
        )
        ask = st.button("Ask the model", type="primary", disabled=not chosen)
        if edited and st.button(
            "Save this wording to the class",
            disabled=not wording_edited,
            help="The class's saved definition and criteria become these; the next run "
            "or label mints a new version. The system text is part of the prompt, not "
            "the class, and is not saved.",
        ):
            try:
                api.update_concept(
                    ontology["id"],
                    leaf["id"],
                    **{field: value or "" for field, value in wording.items()},
                )
                st.session_state.pop("judgement_context", None)
                st.success("Saved to the class.")
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

    if ask:
        todo = [
            payload
            for label in chosen
            for payload in (
                [trial(by_article[label]["document_id"], mine=False)]
                + ([trial(by_article[label]["document_id"], mine=True)] if edited else [])
            )
            if cache_key(payload) not in answers
        ]
        progress = st.progress(0.0, text="Asking…")
        for done, payload in enumerate(todo, start=1):
            try:
                answers[cache_key(payload)] = api.judgement_ask(**payload)
            except ApiError as exc:
                answers[cache_key(payload)] = {"error": exc.detail}
            progress.progress(done / len(todo), text=f"Asked {done} of {len(todo)}")
        progress.empty()

    # ---------------------------------------------------------------------------
    # Answers
    # ---------------------------------------------------------------------------

    def verdict_line(answer: dict | None) -> str:
        if answer is None:
            return "—"
        if answer.get("error"):
            return f"error: {answer['error'][:120]}"
        mark = "✓ match" if answer.get("matched") else "✗ no match"
        confidence = answer.get("confidence")
        return mark + (f" ({confidence:.2f})" if confidence is not None else "")

    st.divider()
    for label in chosen:
        article = by_article[label]
        original = answers.get(cache_key(trial(article["document_id"], mine=False)))
        mine = (
            answers.get(cache_key(trial(article["document_id"], mine=True))) if edited else None
        )
        with st.container(border=True):
            st.markdown(f"**[{article['title'] or article['url']}]({article['url']})**")
            cols = st.columns(4 if edited else 3)
            cols[0].caption("Label")
            cols[0].markdown(
                {True: "✓ match", False: "✗ no match", None: "—"}[article["label"]]
            )
            cols[1].caption(f"Run {run['id']} recorded")
            cols[1].markdown(verdict_line(article["verdict"]))
            cols[2].caption("Original prompt, now")
            cols[2].markdown(verdict_line(original))
            if edited:
                changed = (
                    original is not None
                    and mine is not None
                    and not original.get("error")
                    and not mine.get("error")
                    and original.get("matched") != mine.get("matched")
                )
                cols[3].caption("Your edits" + (" · changed the answer" if changed else ""))
                cols[3].markdown(verdict_line(mine))
            for name, answer in (("Original", original), ("Edited", mine)):
                if answer and answer.get("evidence"):
                    st.caption(f"{name} evidence: “{answer['evidence'][:300]}”")
            if article["verdict"] and article["verdict"].get("evidence"):
                st.caption(f"Run evidence: “{article['verdict']['evidence'][:300]}”")
            with st.expander("The prompt sent" + (" with your edits" if edited else "")):
                try:
                    rendered = (mine or original) or api.judgement_render(
                        **trial(article["document_id"], mine=edited)
                    )
                except ApiError as exc:
                    st.error(exc.detail)
                else:
                    st.caption("System")
                    st.code(rendered["system"], language=None, wrap_lines=True)
                    st.caption("Message")
                    st.code(rendered["prompt"], language=None, wrap_lines=True)


# ---------------------------------------------------------------------------
# Evaluation: a judge version across the runs that used it
# ---------------------------------------------------------------------------


def show_evaluation() -> None:
    try:
        judges = api.judgement_versions(ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not judges:
        st.info("No run of this ontology has judged anything yet.")
        return

    def describe(v: dict) -> str:
        return (
            f"{v['version']} · {v['prompt_id']} · {v['model']} · ontology "
            f"{v['ontology_version']} · {len(v['runs'])} run(s)"
        )

    by_version = {describe(v): v for v in judges}
    judge = by_version[st.selectbox("Judge version", list(by_version), key="ev_judge")]
    st.caption(f"Prompt text fingerprint {judge['prompt']}; provider {judge['provider']}.")
    run_labels = {f"{r['id']} · {r['name']} ({r['status']})": r["id"] for r in judge["runs"]}
    chosen = st.multiselect(
        "Runs pooled",
        list(run_labels),
        default=list(run_labels),
        key=f"ev_runs_{judge['version']}",
    )
    scope = st.radio(
        "Score on",
        ["responsible", "retrieved"],
        format_func=lambda s: {
            "responsible": "Everything this judge was responsible for",
            "retrieved": "Only pairs retrieval selected",
        }[s],
        horizontal=True,
        key="ev_scope",
        help="A per-pair or batched judge answers only what retrieval selected; a "
        "top-down judge answers for every leaf of the articles it processed, true "
        "matches retrieval never offered included. To compare judges of different "
        "styles, score them on what retrieval selected.",
    )
    if not chosen:
        return
    try:
        result = api.judgement_evaluate([run_labels[c] for c in chosen], scope, annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return
    pooled = result["pooled"]
    if not pooled["pairs"]:
        st.info(
            "No labelled pair among what these runs judged. Try a machine annotation set "
            "as truth (sidebar)."
        )
        return

    def interval(name: str) -> str:
        value, ci = pooled[name], pooled[f"{name}_ci"]
        if value is None:
            return "—"
        return (
            f"{value:.2f} ({ci[0]:.2f}–{ci[1]:.2f})"
            if ci and ci[0] is not None
            else f"{value:.2f}"
        )

    cols = st.columns(4)
    cols[0].metric("Precision", interval("precision"))
    cols[1].metric("Recall", interval("recall"))
    cols[2].metric("F1", interval("f1"))
    cols[3].metric("Pairs scored", f"{pooled['pairs']:,}")
    st.caption(
        f"{pooled['tp']} true positives, {pooled['fp']} false positives, {pooled['fn']} "
        f"false negatives over {pooled['documents']} labelled article(s); 95% intervals "
        "from resampling articles. A pair several runs judged counts once, from the "
        "newest run."
    )
    with st.expander("Calibration: is the stated confidence worth anything?"):
        st.caption(
            "Agreement with the labels per confidence band, over the pairs the judge "
            "actually answered. Bars far from the diagonal mean the number is not a "
            "probability, however reasonable it looks."
        )
        st.altair_chart(charts.calibration(result["calibration"]), use_container_width=True)
    st.markdown("**Per run**")
    st.dataframe(
        [
            {
                "Run": f"{r['run_id']} · {r['name']}",
                "Pairs": r["pairs"],
                "Articles": r["documents"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "TP": r["tp"],
                "FP": r["fp"],
                "FN": r["fn"],
            }
            for r in result["runs"]
        ],
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
        },
    )


trials_tab, evaluation_tab = st.tabs(
    ["Try a prompt", "Evaluation"], key="judgement_tab", on_change="rerun"
)
if trials_tab.open:
    with trials_tab:
        show_trials()
if evaluation_tab.open:
    with evaluation_tab:
        show_evaluation()
