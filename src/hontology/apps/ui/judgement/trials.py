"""Try a prompt: a class and a few articles, each asked as the chosen run asks,
with the original prompt and, when edited, with the edits beside it.

Nothing a trial says is stored; answers are kept for the session, by the trial
asked, so asking again costs nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.judgement.answer_card import answer_card
from hontology.apps.ui.judgement.context import Context
from hontology.apps.ui.judgement.prompt_panel import prompt_panel


@dataclass(frozen=True)
class _Asking:
    """What every trial of this class shares; a trial adds the article."""

    run_id: int
    concept_id: int
    prompt_id: str
    system: str
    wording: dict
    own: dict[str, str]  # pasted text, when that is the source

    def trial(self, document_id: int | None, *, mine: bool) -> dict:
        return {
            "run_id": self.run_id,
            "document_id": document_id,
            "concept_id": self.concept_id,
            "prompt_id": self.prompt_id,
            "system": self.system if mine else None,
            "wording": self.wording if mine else None,
            **(
                {"text": self.own["text"], "title": self.own["title"] or None}
                if document_id is None
                else {}
            ),
        }


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def show_trials(ctx: Context) -> None:
    run = ctx.run
    if run is None:
        st.info(
            "Pick a run that has judged something in Settings (top right): a trial is asked as "
            "that run asks, with its model, decoding settings and article text limit, "
            "and its recorded verdicts are shown beside each answer."
        )
        return
    leaves = ctx.api.leaves(ctx.ontology["id"])
    if not leaves:
        st.info("This ontology has no classes yet.")
        return
    by_name = shared.leaf_options(leaves)
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

    prompt_id, system, wording, edited = prompt_panel(ctx, leaf)
    own: dict[str, str] = {}
    answers: dict[str, dict] = st.session_state.setdefault("judgement_answers", {})
    articles = _articles(ctx, run, leaf, source, own)
    if articles is None:
        return
    asking = _Asking(run["id"], leaf["id"], prompt_id, system, wording, own)

    by_article = {_describe(a): a for a in articles}
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
        _ask(ctx, asking, [by_article[label] for label in chosen], edited, answers)

    st.divider()
    for label in chosen:
        article = by_article[label]
        original = answers.get(_cache_key(asking.trial(article["document_id"], mine=False)))
        mine = (
            answers.get(_cache_key(asking.trial(article["document_id"], mine=True)))
            if edited
            else None
        )
        answer_card(
            ctx.api,
            run,
            article,
            original,
            mine,
            edited=edited,
            sent=asking.trial(article["document_id"], mine=edited),
        )


def _articles(
    ctx: Context, run: dict, leaf: dict, source: str, own: dict[str, str]
) -> list[dict] | None:
    """What to ask about: the pasted text, kept in *own*, or the run's articles
    for the class. None when there is nothing to ask about."""
    if source == "Your text":
        title, body = shared.own_text(
            f"The text is sent to this run's judge, {run['provider']} · {run['model']}."
        )
        if not body.strip():
            st.info("Paste an article to ask the model about it.")
            return None
        own.update(text=body, title=title)
        return [
            {
                "document_id": None,
                "url": "",
                "title": title or "Your text",
                "label": None,
                "verdict": None,
            }
        ]
    try:
        articles = ctx.api.judgement_articles(run["id"], leaf["id"], ctx.annotator)
    except ApiError as exc:
        st.error(exc.detail)
        return None
    if not articles:
        st.info(
            f"Run {run['id']} judged no article for this class, and none is "
            "labelled for it. Pick another class or run, or ask about your text."
        )
        return None
    return articles


def _describe(article: dict) -> str:
    label = {True: "label ✓", False: "label ✗", None: "unlabelled"}[article["label"]]
    verdict = article["verdict"]
    said = (
        ""
        if verdict is None
        else " · run "
        + ("✓" if verdict["matched"] else "✗" if verdict["matched"] is False else "error")
    )
    return f"{(article['title'] or article['url'])[:80]} · {label}{said}"


def _ask(
    ctx: Context, asking: _Asking, articles: list[dict], edited: bool, answers: dict
) -> None:
    """Ask every trial of *articles* not answered yet, with a progress bar."""
    todo = [
        payload
        for article in articles
        for payload in (
            [asking.trial(article["document_id"], mine=False)]
            + ([asking.trial(article["document_id"], mine=True)] if edited else [])
        )
        if _cache_key(payload) not in answers
    ]
    progress = st.progress(0.0, text="Asking…")
    for done, payload in enumerate(todo, start=1):
        try:
            answers[_cache_key(payload)] = ctx.api.judgement_ask(**payload)
        except ApiError as exc:
            answers[_cache_key(payload)] = {"error": exc.detail}
        progress.progress(done / len(todo), text=f"Asked {done} of {len(todo)}")
    progress.empty()
