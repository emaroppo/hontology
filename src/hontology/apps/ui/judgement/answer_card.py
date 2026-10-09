"""One article's answers: its label, the run's verdict, the answers now, and the
prompt sent."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import Api, ApiError


def verdict_line(answer: dict | None) -> str:
    if answer is None:
        return "—"
    if answer.get("error"):
        return f"error: {answer['error'][:120]}"
    mark = "✓ match" if answer.get("matched") else "✗ no match"
    confidence = answer.get("confidence")
    return mark + (f" ({confidence:.2f})" if confidence is not None else "")


def answer_card(
    api: Api,
    run: dict,
    article: dict,
    original: dict | None,
    mine: dict | None,
    *,
    edited: bool,
    sent: dict,
) -> None:
    """One article: its label, the run's verdict, the answers now, and the prompt
    sent, rendered from the trial *sent* when no answer carries it."""
    with st.container(border=True):
        if article["url"]:
            st.markdown(f"**[{article['title'] or article['url']}]({article['url']})**")
        else:
            st.markdown(f"**{article['title']}** (pasted)")
        cols = st.columns(4 if edited else 3)
        cols[0].caption("Label")
        cols[0].markdown({True: "✓ match", False: "✗ no match", None: "—"}[article["label"]])
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
                rendered = (mine or original) or api.judgement_render(**sent)
            except ApiError as exc:
                st.error(exc.detail)
            else:
                st.caption("System")
                st.code(rendered["system"], language=None, wrap_lines=True)
                st.caption("Message")
                st.code(rendered["prompt"], language=None, wrap_lines=True)
