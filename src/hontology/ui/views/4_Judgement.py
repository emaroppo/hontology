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

from hontology.ui import charts, shared
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
# The ontology, run and truth chosen in Settings
# ---------------------------------------------------------------------------

ontology = shared.ontology(api, ontologies)
params = shared.current(api)
annotator = params.annotator
runs = api.judgement_runs(ontology["id"])
run = next((r for r in runs if r["id"] == params.run_id), None)
templates = {t["prompt_id"]: t for t in api.judgement_templates()}


def show_trials() -> None:
    if run is None:
        st.info(
            "Pick a run that has judged something in Settings (top right): a trial is asked as "
            "that run asks, with its model, decoding settings and article text limit, "
            "and its recorded verdicts are shown beside each answer."
        )
        return
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
    cols = st.columns([3, 2])
    leaf = by_name[cols[0].selectbox("Class", list(by_name), key="judgement_class")]
    source = cols[1].radio(
        "Ask about",
        ["Your text", "Corpus articles"],
        horizontal=True,
        key="judgement_source",
        help="Your text: an article pasted in, asked about exactly as a corpus "
        "article would be. It is shared with Retrieval's Your text tab.",
    )

    # ---------------------------------------------------------------------------
    # The prompt, in one collapsible section
    # ---------------------------------------------------------------------------

    prompt_ids = list(templates)
    prompt_key = f"judgement_prompt_{run['id']}"
    # The template and the edits are widget values, which Streamlit drops when the
    # page or tab that drew them is left: keep a copy, so they survive a visit
    # elsewhere and are never read missing.
    kept_keys = [prompt_key, "edit_system", *(f"edit_{field}" for field in WORDING)]
    for key in kept_keys:
        if key not in st.session_state and f"keep:{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"keep:{key}"]
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = (
            run["prompt_id"] if run["prompt_id"] in templates else prompt_ids[0]
        )
    prompt_id = st.session_state[prompt_key]
    # Edits start from the template and the class as saved; a new class, prompt or
    # run starts over.
    context = (leaf["id"], prompt_id)
    missing = any(key not in st.session_state for key in kept_keys)
    if st.session_state.get("judgement_context") != context or missing:
        st.session_state["judgement_context"] = context
        st.session_state["edit_system"] = templates[prompt_id]["system"]
        for field in WORDING:
            st.session_state[f"edit_{field}"] = leaf.get(field) or ""

    system = st.session_state["edit_system"]
    wording = {field: st.session_state[f"edit_{field}"].strip() or None for field in WORDING}
    original_wording = {field: (leaf.get(field) or "").strip() or None for field in WORDING}
    system_edited = system != templates[prompt_id]["system"]
    wording_edited = wording != original_wording
    edited = system_edited or wording_edited
    changes = [
        part
        for part, changed in (("system text", system_edited), ("class wording", wording_edited))
        if changed
    ]
    state = f"edited: {' and '.join(changes)}" if changes else "as saved"

    with st.expander(f"**Prompt** · {prompt_id} · {state}", key="judgement_prompt_panel"):
        cols = st.columns([2, 3])
        cols[0].selectbox("Template", prompt_ids, key=prompt_key)
        cols[1].caption(
            f"Asked as run {run['id']}: {run['provider']} · {run['model']}, with its "
            "decoding settings and article text limit."
            + (
                ""
                if run["prompt_id"] in templates
                else f" It used {run['prompt_id']}, which is not a per-pair prompt; its "
                "verdicts are shown for reference, and the trial asks per pair."
            )
        )
        st.text_area("System text", key="edit_system", height=260)
        st.text_area(WORDING["definition"], key="edit_definition", height=80)
        cols = st.columns(2)
        cols[0].text_area(
            WORDING["inclusion_criteria"], key="edit_inclusion_criteria", height=90
        )
        cols[1].text_area(
            WORDING["exclusion_criteria"], key="edit_exclusion_criteria", height=90
        )
        cols = st.columns([1, 2, 3])
        if cols[0].button("Undo my edits", disabled=not edited):
            st.session_state.pop("judgement_context", None)
            st.rerun()
        if cols[1].button(
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
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

    for key in kept_keys:
        st.session_state[f"keep:{key}"] = st.session_state[key]

    own: dict[str, str] = {}  # pasted text, when that is the source

    def trial(document_id: int | None, *, mine: bool) -> dict:
        return {
            "run_id": run["id"],
            "document_id": document_id,
            "concept_id": leaf["id"],
            "prompt_id": prompt_id,
            "system": system if mine else None,
            "wording": wording if mine else None,
            **(
                {"text": own["text"], "title": own["title"] or None}
                if document_id is None
                else {}
            ),
        }

    def cache_key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    answers: dict[str, dict] = st.session_state.setdefault("judgement_answers", {})

    # ---------------------------------------------------------------------------
    # What to ask about
    # ---------------------------------------------------------------------------

    if source == "Your text":
        title, body = shared.own_text(
            f"The text is sent to this run's judge, {run['provider']} · {run['model']}."
        )
        if not body.strip():
            st.info("Paste an article to ask the model about it.")
            return
        own.update(text=body, title=title)
        articles: list[dict] = [
            {
                "document_id": None,
                "url": "",
                "title": title or "Your text",
                "label": None,
                "verdict": None,
            }
        ]
    else:
        try:
            articles = api.judgement_articles(run["id"], leaf["id"], annotator)
        except ApiError as exc:
            st.error(exc.detail)
            return
        if not articles:
            st.info(
                f"Run {run['id']} judged no article for this class, and none is "
                "labelled for it. Pick another class or run, or ask about your text."
            )
            return

    def describe(article: dict) -> str:
        label = {True: "label ✓", False: "label ✗", None: "unlabelled"}[article["label"]]
        verdict = article["verdict"]
        said = (
            ""
            if verdict is None
            else " · run "
            + ("✓" if verdict["matched"] else "✗" if verdict["matched"] is False else "error")
        )
        return f"{(article['title'] or article['url'])[:80]} · {label}{said}"

    by_article = {describe(a): a for a in articles}
    if source == "Your text":
        chosen = list(by_article)
    else:
        chosen = st.multiselect(
            "Articles",
            list(by_article),
            default=list(by_article)[:4],
            key=f"judgement_articles_{leaf['id']}_{run['id']}",
            help="Ordered most informative first: labelled true matches, then "
            "articles the run said match, then labelled articles the run judged.",
        )
    cols = st.columns([1, 4])
    ask = cols[0].button("Ask the model", type="primary", disabled=not chosen)
    cols[1].caption(
        "Each article is asked with the original prompt"
        + (" and with your edits" if edited else "")
        + ". Answers are kept for this session, so asking again costs nothing."
    )

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
            if article["url"]:
                st.markdown(f"**[{article['title'] or article['url']}]({article['url']})**")
            else:
                st.markdown(f"**{article['title']}** (pasted)")
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
            "as truth (Settings, top right)."
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
        st.altair_chart(charts.calibration(result["calibration"]), width="stretch")
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
