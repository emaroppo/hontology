"""`hontology ingest`: pull slices from the news feed, then fetch and tidy articles.

One-shot commands are the baseline. `watch` is the only continuous one, and it
has to be started deliberately — nothing else in the system spawns it.

Commands live in one module per topic and are registered here, in one table,
so `ingest --help` lists them in a deliberate order rather than in import order.
"""

from __future__ import annotations

import typer

from hontology.apps.cli.ingest import articles, calendar, feed

app = typer.Typer(help="Pull slices from the news feed.")

for name, command in (
    ("status", feed.ingest_status),
    ("once", feed.ingest_once),
    ("backfill", feed.ingest_backfill),
    ("themes", feed.ingest_themes),
    ("calendar", calendar.ingest_calendar),
    ("repair-export-loci", feed.ingest_repair_export_loci),
    ("calendar-preview", calendar.ingest_calendar_preview),
    ("scrape", articles.ingest_scrape),
    ("dedup", articles.ingest_dedup),
    ("filter-preview", articles.ingest_filter_preview),
    ("watch", feed.ingest_watch),
):
    app.command(name)(command)
