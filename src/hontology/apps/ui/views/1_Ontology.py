"""Ontology editor: create ontologies, author concepts, import and export.

This is the page that has to work from a completely empty install, because it is
the first thing anyone sees.

Two kinds of ontology, edited differently:

- **Flat** (no ``subclass_of`` edges): a list of event classes, authored entirely
  here. The quickest way to try the pipeline on an event set made up on the spot.
- **Structured** (a class hierarchy): the structure, names and internal classes
  are authored in an OWL editor such as Protégé and arrive by import. This page
  shows the tree and edits only leaf wording, which is what the judge reads.
"""

from __future__ import annotations

import json

import streamlit as st

from hontology.apps.ui import shared
from hontology.apps.ui.client import ApiError

api, ontologies = shared.page("Ontology", "🧭", need_ontology=False)


# ---------------------------------------------------------------------------
# New and imported ontologies; the one shown is the one chosen in Settings
# ---------------------------------------------------------------------------

new_col, import_col = st.columns(2)
with new_col.expander("New ontology", expanded=not ontologies), st.form("create_ontology"):
    new_slug = st.text_input("Slug", placeholder="supply-chain")
    new_name = st.text_input("Name", placeholder="Supply chain disruption")
    description = st.text_area("Description", placeholder="Optional.")
    if st.form_submit_button("Create", type="primary"):
        try:
            created = api.create_ontology(
                slug=new_slug.strip(), name=new_name.strip(), description=description or None
            )
            shared.request("ontology", created["id"])
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)

with import_col.expander("Import", expanded=not ontologies):
    st.caption(
        "OWL (Turtle, `.ttl`) from Protégé, or a JSON export. Re-importing "
        "merges by class name rather than duplicating."
    )
    uploaded = st.file_uploader("File", type=["ttl", "json"])
    reword = st.checkbox(
        "Allow rewording existing classes",
        help="OWL only. Off by default, so a structural edit cannot silently "
        "change the wording labels were made against.",
    )
    if uploaded is not None and st.button("Import"):
        try:
            raw = uploaded.read().decode("utf-8")
            if uploaded.name.endswith(".ttl"):
                imported = api.import_owl(raw, allow_text_change=reword)
            else:
                imported = api.import_ontology(json.loads(raw))
            shared.request("ontology", imported["id"])
            st.rerun()
        except (ApiError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            st.error(str(exc))

params = shared.current(api)
selected = next((o for o in ontologies if o["id"] == params.ontology_id), None)
if selected is None:
    st.info(
        "**Nothing here yet.** Create an ontology above and add a few event classes, "
        "or import one from OWL or JSON."
    )
    st.stop()

ontology_id = selected["id"]
slug = selected["slug"]
tree = api.hierarchy(ontology_id)
classes: list[dict] = tree["classes"]
structured: bool = tree["structured"]


# ---------------------------------------------------------------------------
# Header: counts and versions
# ---------------------------------------------------------------------------

listing = api.versions(ontology_id)
versions: list[dict] = listing["versions"]
current: str | None = listing["current"]
latest = versions[-1]["version"] if versions else None

head = st.columns([3, 1, 1, 1])
head[0].subheader(selected["name"])
head[1].metric("Classes", len(classes))
if structured:
    head[2].metric("Leaves", sum(c["leaf"] for c in classes))
head[3].metric("Version", current or (f"{latest} + edits" if latest else "—"))

# Versions are minted automatically by whatever needs one, so there is nothing
# to press here: the header only says where the live wording stands.
if current is None and classes:
    upcoming = f"v{len(versions) + 1}"
    status = (
        f"Edited since {latest}. The edits become **{upcoming}** automatically "
        "when the next run starts or the next label is saved."
        if latest
        else f"Not versioned yet. It becomes **{upcoming}** automatically when "
        "the first run starts or the first label is saved."
    )
    st.info(status)

caption = (
    "A version pins the wording the model is asked about. Editing a definition or "
    "its criteria (or, in a hierarchy, the structure) makes the next one, and "
    "labels made under the old wording are marked stale. Weights and categories "
    "do not."
)
if versions:
    history = " · ".join(
        f"{v['version']} ({v['n_concepts']} classes, {v['created_at'][:10]})" for v in versions
    )
    caption += f"\n\nVersions: {history}"
st.caption(caption)


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------


def wording_fields(concept: dict) -> dict[str, str]:
    """Definition and criteria, the part of a class the judge reads, as text areas
    in the form being drawn. Returns them stripped, by field."""
    definition = st.text_area(
        "Definition",
        concept.get("definition") or "",
        help="What the class means, in plain language.",
    )
    cols = st.columns(2)
    inclusion = cols[0].text_area(
        "Inclusion criteria",
        concept.get("inclusion_criteria") or "",
        help="What counts. Goes into the judge prompt verbatim.",
    )
    exclusion = cols[1].text_area(
        "Exclusion criteria",
        concept.get("exclusion_criteria") or "",
        help="What does not count — usually the best precision lever.",
    )
    return {
        "definition": definition.strip(),
        "inclusion_criteria": inclusion.strip(),
        "exclusion_criteria": exclusion.strip(),
    }


def wording_form(concept: dict, *, key: str) -> None:
    """The wording alone, in a form of its own."""
    with st.form(key):
        wording = wording_fields(concept)
        if st.form_submit_button("Save wording", type="primary"):
            shared.act(
                api.update_concept, ontology_id, concept["id"], success="Saved.", **wording
            )


# ---------------------------------------------------------------------------
# Structured: tree and class detail
# ---------------------------------------------------------------------------


def show_hierarchy() -> None:
    order = shared.tree_order(classes)
    left, right = st.columns([2, 3])

    with left, st.container(height=640):
        lines = []
        for depth, concept in order:
            name = concept["name"] if concept["leaf"] else f"**{concept['name']}**"
            if len(concept["parents"]) > 1:
                name += " ⧉"
            lines.append(f"{'  ' * depth}- {name}")
        st.markdown("\n".join(lines))
        st.caption("**Bold**: internal class. ⧉: has more than one parent (either, not both).")

    with right:
        first_seen: dict[int, str] = {}
        for depth, concept in order:
            first_seen.setdefault(concept["id"], f"{'· ' * depth}{concept['name']}")
        options = {label: concept_id for concept_id, label in first_seen.items()}
        by_id = {c["id"]: c for c in classes}
        concept = by_id[options[st.selectbox("Class", list(options))]]

        kind = "Leaf" if concept["leaf"] else "Internal class"
        st.markdown(f"### {concept['name']}")
        st.caption(
            " · ".join(
                part
                for part in (
                    kind,
                    f"category: {concept['category']}" if concept["category"] else None,
                    f"weight {concept['weight']:g}" if concept["weight"] is not None else None,
                )
                if part
            )
        )

        facts = {
            "Parents": concept["parents"],
            "Children": concept["children"],
            "Precursor of": concept["precursor_of"],
            "Preceded by": sorted(
                c["name"] for c in classes if concept["name"] in c["precursor_of"]
            ),
            "Groups": concept["groups"],
        }
        for label, values in facts.items():
            if values:
                st.markdown(f"**{label}:** {', '.join(values)}")

        if concept["leaf"]:
            wording_form(concept, key=f"wording_{concept['id']}")
        else:
            for field, label in (
                ("definition", "Definition"),
                ("inclusion_criteria", "Inclusion criteria"),
                ("exclusion_criteria", "Exclusion criteria"),
            ):
                if concept.get(field):
                    st.markdown(f"**{label}.** {concept[field]}")
            st.info(
                "Internal classes, names, categories and the structure itself are "
                "edited in Protégé: download the OWL file from **Export**, edit it, "
                "and import it above. Leaf wording can be edited here."
            )


# ---------------------------------------------------------------------------
# Flat: full authoring
# ---------------------------------------------------------------------------


def show_flat_editor() -> None:
    if not classes:
        st.info("No classes yet. Add some in the next tab.")
    for concept in classes:
        with st.expander(concept["name"]), st.form(f"edit_{concept['id']}"):
            name = st.text_input("Name", concept["name"])
            wording = wording_fields(concept)
            cols = st.columns(2)
            category = cols[0].text_input("Category", concept.get("category") or "")
            weight = cols[1].number_input(
                "Weight",
                value=float(concept.get("weight") or 0.0),
                step=0.5,
                help="Scoring weight. Does not affect versioning or labels.",
            )

            save, remove = st.columns(2)
            if save.form_submit_button("Save", type="primary"):
                shared.act(
                    api.update_concept,
                    ontology_id,
                    concept["id"],
                    success="Saved.",
                    name=name.strip(),
                    **wording,
                    category=category.strip() or None,
                    weight=weight,
                )
            if remove.form_submit_button("Delete"):
                api.delete_concept(ontology_id, concept["id"])
                st.rerun()


def show_add() -> None:
    st.markdown("**Several at once**")
    with st.form("add_many"):
        text = st.text_area(
            "One class per line, as `Name: definition`",
            placeholder=(
                "Port closure: A commercial seaport halts vessel operations.\n"
                "Rail strike: Rail freight workers stop work."
            ),
            height=160,
        )
        if st.form_submit_button("Add all", type="primary"):
            added, errors = 0, []
            for line in text.splitlines():
                if not line.strip():
                    continue
                name, _, definition = line.partition(":")
                try:
                    api.create_concept(
                        ontology_id,
                        name=name.strip(),
                        definition=definition.strip() or None,
                        weight=1.0,
                    )
                    added += 1
                except ApiError as exc:
                    errors.append(f"{name.strip()!r}: {exc.detail}")
            for error in errors:
                st.error(error)
            if added:
                st.success(f"Added {added} class(es). Add criteria to sharpen them.")
                if not errors:
                    st.rerun()

    st.markdown("**One, in full**")
    with st.form("add_concept"):
        name = st.text_input("Name", placeholder="Port closure")
        definition = st.text_area(
            "Definition", placeholder="A commercial seaport halts vessel operations."
        )
        cols = st.columns(2)
        inclusion = cols[0].text_area(
            "Inclusion criteria", placeholder="Closure has already taken effect."
        )
        exclusion = cols[1].text_area(
            "Exclusion criteria",
            placeholder="Threatened or announced closures; routine maintenance.",
        )
        cols = st.columns(2)
        category = cols[0].text_input("Category", placeholder="logistics")
        weight = cols[1].number_input("Weight", value=1.0, step=0.5)

        if st.form_submit_button("Add class", type="primary"):
            shared.act(
                api.create_concept,
                ontology_id,
                success=f"Added {name!r}.",
                name=name.strip(),
                definition=definition.strip() or None,
                inclusion_criteria=inclusion.strip() or None,
                exclusion_criteria=exclusion.strip() or None,
                category=category.strip() or None,
                weight=weight,
            )


# ---------------------------------------------------------------------------
# Health and export: both kinds
# ---------------------------------------------------------------------------


def show_health() -> None:
    st.caption(
        "Checks that catch ontology bugs before they cost labelling hours. "
        "Advisory only — this is your ontology to author."
    )
    try:
        report = api.lint_ontology(ontology_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    cols = st.columns(3)
    cols[0].metric("Classes", report["concepts"])
    cols[1].metric("Warnings", report["warnings"])
    cols[2].metric("Notes", report["info"])

    if not report["findings"]:
        st.success("Nothing flagged.")
    for finding in report["findings"]:
        renderer = st.warning if finding["severity"] == "warning" else st.info
        renderer(
            f"**{finding['concept_name']}** — {finding['check'].replace('_', ' ')}\n\n"
            f"{finding['message']}"
        )


def show_export() -> None:
    st.caption(
        "Keyed by class name throughout, so either file merges cleanly into another "
        "database and is safe to keep in version control next to your labels."
    )
    cols = st.columns(2)
    cols[0].download_button(
        "Download OWL (for Protégé)",
        data=api.export_owl(ontology_id),
        file_name=f"{slug}.ttl",
        mime="text/turtle",
    )
    exported = api.export_ontology(ontology_id)
    cols[1].download_button(
        "Download JSON",
        data=json.dumps(exported, indent=2, ensure_ascii=False),
        file_name=f"{slug}.json",
        mime="application/json",
    )
    st.json(exported, expanded=False)


if structured:
    tabs = st.tabs(["Hierarchy", "Health", "Export"])
    with tabs[0]:
        show_hierarchy()
    with tabs[1]:
        show_health()
    with tabs[2]:
        show_export()
else:
    tabs = st.tabs(["Edit classes", "Add classes", "Health", "Export"])
    with tabs[0]:
        show_flat_editor()
    with tabs[1]:
        show_add()
    with tabs[2]:
        show_health()
    with tabs[3]:
        show_export()
