"""Links: each class's codes, ticked by hand, with candidates from a similarity
run to tick. A similarity run only ranks codes; it never links one."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError
from hontology.apps.ui.filtering.context import SYSTEMS, Context


def code_label(code: dict) -> str:
    """``cameo 1451 Engage in political dissent, riot``; a theme's name only
    repeats its code, so just the usage count it carries is kept."""
    name = code.get("name") or ""
    extra = name[len(code["code"]) :].strip() if name.startswith(code["code"]) else name
    return f"`{code['system']}` **{code['code']}** {extra}".strip()


def class_options(ctx: Context) -> dict[str, int]:
    """Classes in tree order for a hierarchy, by name otherwise, with link counts."""
    options: dict[str, int] = {}
    seen: set[int] = set()
    for depth, concept in shared.tree_order(ctx.classes):
        if concept["id"] in seen:
            continue
        seen.add(concept["id"])
        count = len(ctx.links.get(concept["id"], []))
        options[f"{'· ' * depth}{concept['name']}{f'  ({count})' if count else ''}"] = concept[
            "id"
        ]
    return options


def class_links(ctx: Context, concept: dict) -> None:
    """A class's links, then candidates from the latest run. Ticked means linked."""
    current = {link["code"]["id"]: link for link in ctx.links.get(concept["id"], [])}
    candidates: list[dict] = []
    if ctx.similarity_run is not None:
        try:
            candidates = ctx.api.concept_candidates(
                concept["id"], ctx.similarity_run["run_id"], limit=10
            )
        except ApiError as exc:
            st.error(exc.detail)

    if not current and not candidates:
        st.caption("No codes linked. Run similarity above to see candidates.")
        return

    rows: list[tuple[dict, float | None, dict | None]] = [
        (link["code"], link["score"], link)
        for link in sorted(
            current.values(), key=lambda link: (link["code"]["system"], link["code"]["code"])
        )
    ]
    rows += [
        (c["code"], c["score"], None) for c in candidates if c["code"]["id"] not in current
    ]

    for code, score, link in rows:
        if link is None:
            origin = "candidate"
        elif link["manual"]:
            origin = "linked"
        else:
            # Made by an older version that linked proposals automatically.
            origin = f"proposed by run {link['proposed_by_run']}"
        shown = f" · {score:.3f}" if score is not None else ""
        key = f"code_{concept['id']}_{code['id']}"
        cols = st.columns([6, 1])
        checked = cols[0].checkbox(
            f"{code_label(code)}{shown} · {origin}", value=link is not None, key=key
        )
        if checked != (link is not None):
            shared.act(ctx.api.set_link, concept["id"], code["id"], linked=checked)
        proposal = link is not None and not link["manual"]
        if proposal and cols[1].button(
            "Confirm", key=f"keep_{key}", help="Make it a hand-made link."
        ):
            shared.act(ctx.api.set_link, concept["id"], code["id"], linked=True)
    if ctx.similarity_run is not None:
        st.caption(
            f"Candidates are the closest {SYSTEMS[ctx.similarity_run['system']]} in "
            f"similarity run {ctx.similarity_run['run_id']}."
        )


def similarity_controls(ctx: Context) -> None:
    loaded = [slug for slug in SYSTEMS if slug in ctx.systems]
    if not loaded:
        st.info("Load a codebook in the **Codebooks** tab first.")
        return
    with st.expander("Propose candidates by similarity", expanded=ctx.similarity_run is None):
        st.caption(
            "Embeds every class and every candidate code and ranks codes by "
            "closeness. Nothing is linked: the closest codes appear under each "
            "class to tick. For themes only event-like ones are candidates: those "
            "used at least 10,000 times, without the entity lists (occupations, "
            "languages, species)."
        )
        cols = st.columns([2, 2, 1])
        slug = cols[0].selectbox(
            "Code system", loaded, format_func=lambda s: SYSTEMS[s], key="similarity_system"
        )
        levels = [level["level"] for level in ctx.systems[slug]["levels"]]
        level = cols[1].selectbox("Level", levels, key=f"similarity_level_{slug}")
        cols[2].markdown("&nbsp;")
        if cols[2].button("Run", type="primary"):
            try:
                with st.spinner("Embedding and scoring…"):
                    result = ctx.api.run_similarity(
                        ontology_id=ctx.ontology_id, system=slug, level=level
                    )
            except ApiError as exc:
                st.error(exc.detail)
                return
            runs = st.session_state.setdefault("similarity_runs", {})
            runs[ctx.ontology_id] = {"run_id": result["run_id"], "system": slug}
            st.session_state["filtering_message"] = (
                f"Similarity run {result['run_id']}: {result['n_scores']:,} scores. "
                "Each class now lists its closest codes to tick."
            )
            st.rerun()


def show_links(ctx: Context) -> None:
    similarity_controls(ctx)
    options = class_options(ctx)
    concept_id = options[st.selectbox("Class", list(options), key="filter_class")]
    concept = next(c for c in ctx.classes if c["id"] == concept_id)
    if concept.get("definition"):
        st.caption(concept["definition"])
    class_links(ctx, concept)

    unlinked = sorted(c["name"] for c in ctx.classes if c["id"] not in ctx.links)
    if ctx.links and unlinked:
        with st.expander(f"Classes with no links ({len(unlinked)})"):
            st.caption(
                "The filter can never admit an article for these on their own: "
                "only articles some other class's codes let in reach them."
            )
            st.markdown(", ".join(unlinked))
