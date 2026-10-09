"""Labelling: where ground truth gets made, in two ways.

**Sample** labels a frozen random sample a whole document at a time. A labelled
sample is what arms are scored against, so it is labelled blind: this tab shows
the article and the leaves, never a run's verdicts, and not the calendar stratum
the document was drawn from, which names the event it was expected to report.
Documents come in the sample's frozen order, so whatever prefix is done is itself
a random sample. Saving answers every leaf at once: the ticked ones apply, the
rest do not.

**Queue** labels single pairs, ranked by how much a label would teach, and
adjudicates machine proposals. It has to answer two questions fast: *why is this
pair in front of me*, and *what did the models say*. Both are shown, because a
queue that hides its reasoning trains you to rubber stamp it — and a
rubber-stamped machine proposal is exactly the label that poisons the metrics it
is supposed to validate.

Sample comes first so the page opens blind: the queue shows run verdicts, and
only the open tab is run at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from hontology.ui import shared
from hontology.ui.client import ApiError

api, ontologies = shared.page("Labelling", "🏷️")


def sample_tab() -> None:
    ontology = shared.ontology(api, ontologies)
    manifest_path = shared.current(api).sample
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read the sample manifest (Settings, top right): {exc}")
        return
    # A sample is labelled against the ontology of the run it was drawn from.
    try:
        drawn_from = api.get_run(manifest["run_id"])["ontology_id"]
    except (ApiError, KeyError):
        drawn_from = None
    if drawn_from is not None and drawn_from != ontology["id"]:
        st.warning(
            f"This sample was drawn from run {manifest['run_id']}, of another ontology. "
            "Pick that ontology in Settings (top right) to label it."
        )
        return
    first = st.number_input(
        "Label the first", min_value=1, value=120, step=10, key="sample_first"
    )

    document_ids = [entry["document_id"] for entry in manifest["order"]][: int(first)]
    try:
        statuses = api.documents_status(ontology["id"], document_ids)
        leaves = api.leaves(ontology["id"])
    except ApiError as exc:
        st.error(exc.detail)
        return

    done = sum(1 for s in statuses if s["labelled"])
    st.progress(done / len(statuses), text=f"{done} of {len(statuses)} labelled")

    # Start at the first document not yet labelled; navigation moves from there.
    if st.session_state.get("sample_manifest") != (manifest_path, int(first)):
        st.session_state["sample_manifest"] = (manifest_path, int(first))
        st.session_state.pop("sample_position", None)
    if "sample_position" not in st.session_state:
        st.session_state["sample_position"] = next(
            (i for i, s in enumerate(statuses) if not s["labelled"]), 0
        )

    def go(position: int) -> None:
        st.session_state["sample_position"] = max(0, min(position, len(statuses) - 1))

    position = st.session_state["sample_position"]
    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("← Previous", disabled=position == 0):
        go(position - 1)
        st.rerun()
    if nav[1].button("Next →", disabled=position == len(statuses) - 1):
        go(position + 1)
        st.rerun()
    if nav[2].button("Next unlabelled", disabled=done == len(statuses)):
        later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
        earlier = [i for i, s in enumerate(statuses) if not s["labelled"]]
        go((later or earlier)[0])
        st.rerun()
    jump = nav[3].number_input(
        "Go to position", min_value=1, max_value=len(statuses), value=position + 1
    )
    if jump - 1 != position:
        go(jump - 1)
        st.rerun()

    try:
        document = api.document_for_labelling(ontology["id"], document_ids[position])
    except ApiError as exc:
        st.error(exc.detail)
        return

    text_column, label_column = st.columns([3, 2])

    with text_column:
        state = "labelled" if document["labelled"] else "not labelled yet"
        st.caption(f"Position {position + 1} of {len(statuses)} · {state}")
        # Many scraped articles carry no title; the text's first line usually is one.
        first_line = (document["body"] or "").strip().split("\n", 1)[0][:160]
        st.subheader(document["title"] or first_line or document["url"])
        st.markdown(f"[{document['url']}]({document['url']})")
        with st.container(height=640):
            if document["body"]:
                st.text(document["body"])
            else:
                st.warning("No text stored for this article. Read it at the link above.")

    with label_column:
        st.markdown("**Which leaves does the article report?** Untick all for none.")
        current = set(document["positives"])
        key = document["document_id"]
        with st.form(f"labels_{key}"):
            chosen: list[int] = []
            family = None
            for leaf in leaves:
                if leaf["families"][0] != family:
                    family = leaf["families"][0]
                    st.markdown(f"**{family}**")
                explanation = "\n\n".join(
                    part
                    for part in (
                        leaf["definition"],
                        leaf["inclusion_criteria"]
                        and f"Counts when: {leaf['inclusion_criteria']}",
                        leaf["exclusion_criteria"]
                        and f"Does not count when: {leaf['exclusion_criteria']}",
                    )
                    if part
                )
                if st.checkbox(
                    leaf["name"],
                    value=leaf["id"] in current,
                    key=f"{key}_{leaf['id']}",
                    help=explanation,
                ):
                    chosen.append(leaf["id"])
            note = st.text_area("Note (optional)", value=document["note"] or "")
            saved = st.form_submit_button("Save and next", type="primary")

        if saved:
            try:
                api.label_document(ontology["id"], key, chosen, note)
            except ApiError as exc:
                st.error(exc.detail)
                return
            later = [i for i, s in enumerate(statuses) if not s["labelled"] and i > position]
            go(later[0] if later else position + 1)
            st.rerun()


def queue_tab() -> None:
    ontology = shared.ontology(api, ontologies)
    cols = st.columns(3)
    limit = cols[0].slider("Queue size", 5, 100, 25, 5)
    per_concept_cap = cols[1].number_input(
        "Max per concept",
        min_value=0,
        value=5,
        help="0 means no cap. A cap stops the noisiest concept from filling the queue.",
    )
    include_unjudged = cols[2].toggle(
        "Include never-judged pairs",
        value=True,
        help="Retrieved but never judged. Nothing else surfaces these.",
    )

    with st.expander("Move the bank"):
        st.caption("Keyed by document URL and concept name, so it travels between databases.")
        try:
            st.download_button(
                "Download labels (CSV)",
                data=api.export_labels(ontology["id"]),
                file_name=f"labels-{ontology['slug']}.csv",
                mime="text/csv",
            )
        except Exception as exc:  # noqa: BLE001 - a download button must not break the page
            st.caption(f"export unavailable: {exc}")

        uploaded = st.file_uploader("Import labels (CSV)", type="csv")
        overwrite = st.toggle(
            "Overwrite existing",
            value=False,
            help="Off by default so an import cannot silently destroy adjudicated work.",
        )
        if uploaded is not None and st.button("Import"):
            try:
                report = api.import_labels(
                    ontology["id"],
                    uploaded.read().decode("utf-8"),
                    overwrite=overwrite,
                )
                st.success(
                    f"created {report['created']}, updated {report['updated']}, "
                    f"skipped {report['skipped_existing']} existing"
                )
                if report["unknown_concepts"]:
                    st.warning(
                        "skipped unknown concept(s): " + ", ".join(report["unknown_concepts"])
                    )
                if report["bad_row_count"]:
                    st.warning(f"{report['bad_row_count']} unreadable row(s)")
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)

    stats = api.label_stats(ontology["id"])

    row = st.columns(5)
    row[0].metric("Labels", stats["total"])
    row[1].metric("Counted", stats["trusted"], help="Human, adjudicated or imported.")
    row[2].metric(
        "Awaiting review",
        stats["pending_adjudication"],
        help="Machine proposals. These do NOT count until a human confirms them.",
    )
    row[3].metric(
        "Stale",
        stats["stale"],
        help="The concept was reworded after the label was made, so it answers a "
        "question no longer being asked. Excluded from metrics by default.",
    )
    row[4].metric("Observations", stats.get("observations", 0))

    if stats["stale"]:
        st.warning(
            f"{stats['stale']} label(s) went stale after a concept was edited. They are "
            "excluded from metrics until re-adjudicated — confirming one against the "
            "current wording makes it count again."
        )

    st.divider()

    def adjudication_panel(rows: list[dict], *, kind: str) -> None:
        """Confirm or flip a proposal. This is the step that makes a label count."""
        for row in rows:
            with st.container(border=True):
                head = st.columns([3, 1])
                title = row["document_title"] or row["document_url"][:64]
                head[0].markdown(
                    f"**{row['concept_name']}** · [{title}]({row['document_url']})"
                )
                proposed = "matched" if row["proposed_matched"] else "no match"
                by = row.get("proposed_by") or "a model"
                head[1].markdown(f"proposed: **{proposed}**")
                head[1].caption(f"by {by} · {row.get('ontology_version') or '—'}")

                if row.get("note"):
                    st.caption(f"note: {row['note']}")

                buttons = st.columns([1, 1, 6])
                key = f"{kind}_{row['id']}"
                for col, label, prefix, matched, primary in (
                    (buttons[0], "Confirm", "ok", row["proposed_matched"], True),
                    (buttons[1], "Flip", "flip", not row["proposed_matched"], False),
                ):
                    if col.button(
                        label, key=f"{prefix}_{key}", type="primary" if primary else "secondary"
                    ):
                        shared.act(api.adjudicate_label, row["id"], matched=matched)

    if stats["pending_adjudication"]:
        st.subheader(f"Awaiting review ({stats['pending_adjudication']})")
        st.caption(
            "Machine proposals. They do **not** count as ground truth until confirmed "
            "or flipped here — scoring a model against another model's unreviewed "
            "labels measures agreement, not correctness. Disagreeing is the most "
            "valuable outcome, not an error."
        )
        try:
            adjudication_panel(
                api.pending_adjudication(ontology["id"], limit=25), kind="pending"
            )
        except ApiError as exc:
            st.error(exc.detail)
        st.divider()

    if stats["stale"]:
        with st.expander(f"Stale labels ({stats['stale']}) — re-adjudicate to restore them"):
            st.caption(
                "The concept was reworded after these were made, so they answer a "
                "question no longer being asked. Confirming against the current wording "
                "makes them count again."
            )
            try:
                adjudication_panel(api.stale_detail(ontology["id"], limit=25), kind="stale")
            except ApiError as exc:
                st.error(exc.detail)
        st.divider()

    st.subheader("Queue")

    try:
        items = api.labelling_queue(
            ontology["id"],
            limit=limit,
            per_concept_cap=int(per_concept_cap) or None,
            include_unjudged=include_unjudged,
        )
    except ApiError as exc:
        st.error(exc.detail)
        return

    if not items:
        st.success(
            "Nothing queued. Either everything retrieved has been labelled, or no runs "
            "have been executed yet."
        )
        return

    REASON_HELP = {
        "runs disagree": "Two runs split on this pair, so at least one is wrong. "
        "This needs no model to be calibrated, which is why it ranks first.",
        "model uncertain": "A run whose confidence carries real signal sat near the "
        "decision boundary here.",
        "never judged": "Retrieved but never judged by any run — a blind spot nothing "
        "else surfaces.",
        "coverage": "This concept has few labels relative to the others.",
    }

    st.caption(f"{len(items)} pair(s) queued, most informative first.")

    for item in items:
        with st.container(border=True):
            head = st.columns([3, 1])
            title = item["document_title"] or item["document_url"][:70]
            head[0].markdown(f"**{item['concept_name']}** · [{title}]({item['document_url']})")
            head[1].markdown(f"`{item['reason']}`")
            head[1].caption(REASON_HELP.get(item["reason"], ""))

            if item["verdicts"]:
                for verdict in item["verdicts"]:
                    mark = "matched" if verdict["matched"] else "no match"
                    confidence = (
                        f"{verdict['confidence']:.2f}"
                        if verdict["confidence"] is not None
                        else "—"
                    )
                    trust = (
                        ""
                        if verdict["confidence_usable"]
                        else "  ·  confidence not informative"
                    )
                    st.markdown(
                        f"&nbsp;&nbsp;run {verdict['run_id']}: **{mark}** "
                        f"(confidence {confidence}){trust}",
                        unsafe_allow_html=True,
                    )
                    if verdict.get("evidence"):
                        st.caption(f"“{verdict['evidence'][:220]}”")
            else:
                st.caption("No run has judged this pair.")

            note = st.text_input(
                "Note (optional)", key=f"note_{item['document_id']}_{item['concept_id']}"
            )
            buttons = st.columns([1, 1, 6])
            key = f"{item['document_id']}_{item['concept_id']}"
            for col, label, prefix, matched, primary in (
                (buttons[0], "Matches", "yes", True, True),
                (buttons[1], "Does not", "no", False, False),
            ):
                if col.button(
                    label, key=f"{prefix}_{key}", type="primary" if primary else "secondary"
                ):
                    shared.act(
                        api.create_label,
                        document_id=item["document_id"],
                        concept_id=item["concept_id"],
                        matched=matched,
                        note=note or None,
                    )


shared.lazy_tabs(
    "labelling_tab",
    {"Sample (whole documents, blind)": sample_tab, "Queue (pairs and review)": queue_tab},
)
