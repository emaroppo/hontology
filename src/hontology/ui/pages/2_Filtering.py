"""Filtering: which feed articles are worth downloading for an ontology.

The GDELT feeds tag every article before anything is fetched: the event export
with CAMEO event codes, the knowledge graph with GKG themes. Linking a class to
codes lets the optional pre-download filter fetch only articles whose codes link
to some class. That gate bounds recall for everything downstream, so this page
puts the three things needed to set it well side by side: the links, what they
would keep, and what each code has been worth once articles were labelled.

Every link is ticked by hand. A similarity run only ranks codes against each
class, so its proposals appear as candidates; proposing links automatically let
far too much through.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from hontology.ui import truth
from hontology.ui.client import Api, ApiError

st.set_page_config(page_title="Filtering", page_icon="🧹", layout="wide")

api = Api()

st.title("🧹 Filtering")
st.caption(
    "Which feed articles are worth downloading. Classes link to the codes the feeds "
    "tag articles with (CAMEO event codes, GKG themes), and the optional "
    "pre-download filter fetches only articles whose codes link to some class. "
    "Semantic retrieval does not use the links, so an ontology can do without any."
)

if not api.healthy():
    st.error(f"The API is not reachable at `{api.base_url}`. Start it with `make api`.")
    st.stop()

ontologies = api.list_ontologies()
if not ontologies:
    st.info("No ontologies yet. Create one on the **Ontology** page first.")
    st.stop()

with st.sidebar:
    by_label = {f"{o['name']} ({o['slug']})": o for o in ontologies}
    ontology = by_label[st.selectbox("Ontology", list(by_label), key="filter_ontology")]
ontology_id = ontology["id"]

SYSTEMS = {"cameo": "CAMEO event codes", "gkg-themes": "GKG themes"}

tree = api.hierarchy(ontology_id)
classes: list[dict] = tree["classes"]
links = api.ontology_links(ontology_id)
systems = {s["slug"]: s for s in api.code_systems()}
# The latest similarity run this session, whose scores are the candidates.
similarity_run: dict | None = st.session_state.get("similarity_runs", {}).get(ontology_id)
# What the links are now, so a preview or report computed before a tick can say
# it is out of date.
fingerprint = tuple(
    sorted((c, link["code"]["id"]) for c, rows in links.items() for link in rows)
)

if not classes:
    st.info("This ontology has no classes yet. Add some on the **Ontology** page.")
    st.stop()

linked_by_system: dict[str, int] = {}
for rows in links.values():
    for link in rows:
        system = link["code"]["system"]
        linked_by_system[system] = linked_by_system.get(system, 0) + 1

head = st.columns(4)
head[0].metric("Classes with links", f"{len(links)} of {len(classes)}")
head[1].metric("CAMEO links", linked_by_system.get("cameo", 0))
head[2].metric("Theme links", linked_by_system.get("gkg-themes", 0))
head[3].metric("Filter", "usable" if links else "no links yet")

message = st.session_state.pop("filtering_message", None)
if message:
    st.success(message)


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def code_label(code: dict) -> str:
    """``cameo 1451 Engage in political dissent, riot``; a theme's name only
    repeats its code, so just the usage count it carries is kept."""
    name = code.get("name") or ""
    extra = name[len(code["code"]) :].strip() if name.startswith(code["code"]) else name
    return f"`{code['system']}` **{code['code']}** {extra}".strip()


def class_options() -> dict[str, int]:
    """Classes in tree order for a hierarchy, by name otherwise, with link counts."""
    by_name = {c["name"]: c for c in classes}
    order: list[tuple[int, dict]] = []

    def visit(concept: dict, depth: int) -> None:
        order.append((depth, concept))
        for child in concept["children"]:
            visit(by_name[child], depth + 1)

    for concept in classes:
        if not concept["parents"]:
            visit(concept, 0)
    options: dict[str, int] = {}
    seen: set[int] = set()
    for depth, concept in order:
        if concept["id"] in seen:
            continue
        seen.add(concept["id"])
        count = len(links.get(concept["id"], []))
        options[f"{'· ' * depth}{concept['name']}{f'  ({count})' if count else ''}"] = concept[
            "id"
        ]
    return options


def class_links(concept: dict) -> None:
    """A class's links, then candidates from the latest run. Ticked means linked."""
    current = {link["code"]["id"]: link for link in links.get(concept["id"], [])}
    candidates: list[dict] = []
    if similarity_run is not None:
        try:
            candidates = api.concept_candidates(
                concept["id"], similarity_run["run_id"], limit=10
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
            try:
                api.set_link(concept["id"], code["id"], linked=checked)
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
        proposal = link is not None and not link["manual"]
        if proposal and cols[1].button(
            "Confirm", key=f"keep_{key}", help="Make it a hand-made link."
        ):
            try:
                api.set_link(concept["id"], code["id"], linked=True)
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
    if similarity_run is not None:
        st.caption(
            f"Candidates are the closest {SYSTEMS[similarity_run['system']]} in "
            f"similarity run {similarity_run['run_id']}."
        )


def similarity_controls() -> None:
    loaded = [slug for slug in SYSTEMS if slug in systems]
    if not loaded:
        st.info("Load a codebook in the **Codebooks** tab first.")
        return
    with st.expander("Propose candidates by similarity", expanded=similarity_run is None):
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
        levels = [level["level"] for level in systems[slug]["levels"]]
        level = cols[1].selectbox("Level", levels, key=f"similarity_level_{slug}")
        cols[2].markdown("&nbsp;")
        if cols[2].button("Run", type="primary"):
            try:
                with st.spinner("Embedding and scoring…"):
                    result = api.run_similarity(
                        ontology_id=ontology_id, system=slug, level=level
                    )
            except ApiError as exc:
                st.error(exc.detail)
                return
            runs = st.session_state.setdefault("similarity_runs", {})
            runs[ontology_id] = {"run_id": result["run_id"], "system": slug}
            st.session_state["filtering_message"] = (
                f"Similarity run {result['run_id']}: {result['n_scores']:,} scores. "
                "Each class now lists its closest codes to tick."
            )
            st.rerun()


def show_links() -> None:
    similarity_controls()
    options = class_options()
    concept_id = options[st.selectbox("Class", list(options), key="filter_class")]
    concept = next(c for c in classes if c["id"] == concept_id)
    if concept.get("definition"):
        st.caption(concept["definition"])
    class_links(concept)

    unlinked = sorted(c["name"] for c in classes if c["id"] not in links)
    if links and unlinked:
        with st.expander(f"Classes with no links ({len(unlinked)})"):
            st.caption(
                "The filter can never admit an article for these on their own: "
                "only articles some other class's codes let in reach them."
            )
            st.markdown(", ".join(unlinked))


# ---------------------------------------------------------------------------
# Preview and per-code report: slow, so computed on request and kept
# ---------------------------------------------------------------------------


def cached(kind: str) -> dict | None:
    return st.session_state.get("filter_cache", {}).get((ontology_id, kind))


def compute(kind: str, fetch) -> None:
    try:
        with st.spinner("Walking every feed record; this takes a few minutes…"):
            data = fetch(ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.session_state.setdefault("filter_cache", {})[(ontology_id, kind)] = {
        "data": data,
        "at": datetime.now().strftime("%H:%M"),
        "fingerprint": fingerprint,
    }
    st.rerun()


def freshness(entry: dict) -> None:
    if entry["fingerprint"] != fingerprint:
        st.warning(f"Computed at {entry['at']}, before the links last changed. Recompute.")
    else:
        st.caption(f"Computed at {entry['at']}.")


def show_preview() -> None:
    st.caption(
        "What the filter would do to the articles not downloaded yet, which is "
        "what the download budget is spent on. Nothing is fetched."
    )
    entry = cached("preview")
    if st.button("Recompute" if entry else "Compute", key="preview_go", type="primary"):
        compute("preview", api.filter_preview)
    if entry is None:
        return
    freshness(entry)
    data = entry["data"]
    if not data["usable"]:
        st.info(data["reason"])
        return
    cols = st.columns(4)
    cols[0].metric("Share kept", f"{data['share_kept']:.1%}")
    cols[1].metric("Would download", f"{data['unfetched_matching']:,}")
    cols[2].metric("Would skip", f"{data['unfetched_skipped']:,}")
    cols[3].metric("Linked codes", data["linked_codes"])
    st.caption(
        f"Of {data['documents_total']:,} feed articles, {data['documents_matching']:,} "
        f"match some link; {data['documents_unfetched']:,} are not downloaded yet."
    )


# ---------------------------------------------------------------------------
# Codebooks
# ---------------------------------------------------------------------------


def show_codebooks() -> None:
    ingest = {"cameo": api.ingest_cameo, "gkg-themes": api.ingest_themes}
    cols = st.columns(len(SYSTEMS))
    for col, (slug, title) in zip(cols, SYSTEMS.items(), strict=True):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            system = systems.get(slug)
            if system is None:
                st.caption("Not loaded.")
            else:
                n_codes = sum(level["n_codes"] for level in system["levels"])
                levels = ", ".join(level["level"] for level in system["levels"])
                st.caption(f"{n_codes:,} codes ({levels})")
            label = "Reload" if system else "Load"
            if st.button(label, key=f"ingest_{slug}", help="Fetches the public lookup."):
                try:
                    with st.spinner("Fetching…"):
                        result = ingest[slug]()
                    st.session_state["filtering_message"] = (
                        f"{title}: {result.get('total', '?'):,} codes loaded."
                    )
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.detail)


# ---------------------------------------------------------------------------
# Evaluation: a version of the links, link by link
# ---------------------------------------------------------------------------


def show_evaluation() -> None:
    try:
        listing = api.filtering_versions(ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    live = listing["live"]
    options: dict[str, str | None] = {
        f"Live links ({live['n_links']})"
        + (f", the same as {live['is']}" if live["is"] else ""): None
    }
    for snap in listing["snapshots"]:
        used = f", used by runs {', '.join(map(str, snap['runs']))}" if snap["runs"] else ""
        options[
            f"{snap['version']} ({snap['n_links']} links, {snap['created_at'][:10]}{used})"
        ] = snap["version"]
    cols = st.columns([3, 1])
    version = options[cols[0].selectbox("Links", list(options), key="fe_version")]
    unkept = version is None and not live["is"] and live["n_links"]
    if unkept and cols[1].button(
        "Keep as a version", help="Snapshot the live links to compare later."
    ):
        try:
            kept = api.filtering_snapshot(ontology_id)
            st.session_state["filtering_message"] = f"Kept the live links as {kept['version']}."
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)

    annotator = truth.pick(api, ontology_id, st, key="fe_truth")
    payload = {"ontology_id": ontology_id, "version": version, "annotator": annotator}
    try:
        labels = api.filtering_evaluate("labels", **payload)
    except ApiError as exc:
        st.error(exc.detail)
        return

    st.markdown("**Against the labels**")
    cols = st.columns(3)
    cols[0].metric(
        "True matches admitted",
        f"{labels['positives_admitted']} of {labels['positives']}",
    )
    cols[1].metric(
        "Labelled articles admitted", f"{labels['documents_admitted']} of {labels['documents']}"
    )
    st.caption(
        "Labelled articles were mostly downloaded because the filter admitted them, so "
        "these counts flatter it, its misses most of all. The calendar below does not "
        "have that bias."
    )

    key = (ontology_id, version)
    cache = st.session_state.setdefault("filter_eval", {})
    st.markdown("**Against the calendar and over the whole corpus** (a few minutes each)")
    cols = st.columns([2, 1, 1])
    calendar_path = cols[0].text_input(
        "Calendar", value="data/calendars/supply-chain-v2-small.csv", key="fe_calendar"
    )
    if cols[1].button("Score on the calendar"):
        try:
            with st.spinner("Walking every calendar window…"):
                cache[(key, "calendar")] = api.filtering_evaluate(
                    "calendar", **payload, calendar_path=calendar_path
                )
        except ApiError as exc:
            st.error(exc.detail)
    if cols[2].button("Count the cost"):
        try:
            with st.spinner("Walking every feed record…"):
                cache[(key, "cost")] = api.filtering_evaluate("cost", **payload)
        except ApiError as exc:
            st.error(exc.detail)
    calendar_result = cache.get((key, "calendar"))
    cost = cache.get((key, "cost"))

    if calendar_result:
        st.dataframe(
            [
                {
                    "Kind": kind,
                    "Events": c["events"],
                    "With articles in the window": c["observable"],
                    "Reached by any link": c["reached"],
                    "Reached by the event's own class": c["reached_by_own_class"],
                }
                for kind, c in calendar_result["by_kind"].items()
            ],
            hide_index=True,
        )
        missed = calendar_result["not_reached_by_own_class"]
        if missed:
            with st.expander(f"Events the own class's links miss ({len(missed)})"):
                for event in missed:
                    note = "" if event["known_class"] else " (no class of that name)"
                    st.markdown(
                        f"- {event['kind']} · {event['class']}{note}: {event['description']}"
                    )

    st.markdown("**Link by link**")
    st.caption(
        "Against the labels for the class each link reaches: TP admitted true matches, "
        "FP admitted non-matches, FN true matches it does not admit, TN non-matches it "
        "rightly keeps out; unique TP, true matches no other link admits. Corpus columns "
        "count feed articles per code: admitted, and admitted by no other link, which "
        "removing the link would stop downloading."
    )
    rows = []
    for link in labels["links"]:
        code_key = f"{link['system']}:{link['code']}"
        row = {
            "Class": link["class"],
            "System": link["system"],
            "Code": link["code"],
            "TP": link["tp"],
            "FP": link["fp"],
            "FN": link["fn"],
            "TN": link["tn"],
            "Unique TP": link["unique_tp"],
        }
        if calendar_result:
            row["Calendar events"] = len(calendar_result["per_code"].get(code_key, []))
        if cost:
            counts = cost["codes"].get(code_key, {})
            row["Corpus admitted"] = counts.get("admitted", 0)
            row["Only this code"] = counts.get("only_this", 0)
            row["Downloaded"] = counts.get("downloaded", 0)
        rows.append(row)
    rows.sort(key=lambda r: (-r.get("Only this code", 0), -r["FP"]))
    st.dataframe(rows, hide_index=True)


links_tab, preview_tab, evaluation_tab, codebooks_tab = st.tabs(
    ["Links", "What it keeps", "Evaluation", "Codebooks"],
    key="filtering_tab",
    on_change="rerun",
)
if links_tab.open:
    with links_tab:
        show_links()
if preview_tab.open:
    with preview_tab:
        show_preview()
if evaluation_tab.open:
    with evaluation_tab:
        show_evaluation()
if codebooks_tab.open:
    with codebooks_tab:
        show_codebooks()
