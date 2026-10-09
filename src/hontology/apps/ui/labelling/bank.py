"""Move the bank: the labels exported and imported as CSV, keyed by document URL
and concept name so they travel between databases."""

from __future__ import annotations

import streamlit as st

from hontology.apps.ui.client import Api, ApiError


def move_the_bank(api: Api, ontology: dict) -> None:
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
