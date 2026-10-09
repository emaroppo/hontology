"""Command line entry point."""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path

import typer
from sqlalchemy import select, text

from hontology.config import get_settings
from hontology.db.session import session_scope
from hontology.ingest import gdelt, scrape, service
from hontology.judge import run as judge_run
from hontology.judge.providers.base import ProviderError
from hontology.ontology import service as ontology_service
from hontology.ontology import snapshots

app = typer.Typer(help="hontology — ontology-driven event detection and evaluation.")
ontology_app = typer.Typer(help="Manage ontologies.")
ingest_app = typer.Typer(help="Pull slices from the news feed.")
# Documents fetched per committed batch by `ingest scrape`.
SCRAPE_BATCH = 200
app.add_typer(ontology_app, name="ontology")
run_app = typer.Typer(help="Configure and execute detection runs.")
app.add_typer(ingest_app, name="ingest")
eval_app = typer.Typer(help="Score runs against the ground-truth bank.")
app.add_typer(run_app, name="run")
labels_app = typer.Typer(help="Move the ground-truth bank in and out.")
app.add_typer(eval_app, name="eval")
app.add_typer(labels_app, name="labels")


@app.command()
def doctor() -> None:
    """Check that the things this project depends on are reachable."""
    settings = get_settings()
    ok = True

    typer.echo(f"data dir       {settings.data_dir.resolve()}")

    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        typer.echo(f"database       ok  ({settings.psycopg_url()})")
    except Exception as exc:  # noqa: BLE001 - a diagnostic should never crash
        ok = False
        typer.secho(f"database       FAIL  {exc}", fg=typer.colors.RED)

    try:
        provider = judge_run.get_provider(settings.default_judge_provider)
        typer.echo(f"llm provider   {provider.health()}")
    except ProviderError as exc:
        # Not fatal: the ontology layer works fine without a model.
        typer.secho(f"llm provider   unavailable  ({exc})", fg=typer.colors.YELLOW)

    raise typer.Exit(0 if ok else 1)


@ontology_app.command("list")
def ontology_list() -> None:
    """List ontologies and their concept counts."""
    with session_scope() as session:
        rows = ontology_service.list_ontologies(session)
        if not rows:
            typer.echo("(none — this install is empty by design)")
            return
        for ontology in rows:
            n = len(ontology_service.list_concepts(session, ontology.id))
            typer.echo(f"{ontology.id:>4}  {ontology.slug:<24} {ontology.name}  [{n} concepts]")


@ontology_app.command("import")
def ontology_import(path: Path) -> None:
    """Create or merge an ontology from a JSON export."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    with session_scope() as session:
        ontology = ontology_service.import_ontology(session, payload)
        session.flush()
        typer.echo(f"imported {ontology.slug!r} (id {ontology.id})")


@ontology_app.command("export")
def ontology_export(ontology_id: int, out: Path | None = None) -> None:
    """Write an ontology to a portable JSON file, or stdout."""
    with session_scope() as session:
        payload = ontology_service.export_ontology(session, ontology_id)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")


@ontology_app.command("export-owl")
def ontology_export_owl(ontology_id: int, out: Path) -> None:
    """Write the ontology as OWL (Turtle), for Protégé or any OWL tool."""
    from hontology.ontology import owl

    with session_scope() as session:
        out.write_text(owl.export_turtle(session, ontology_id), encoding="utf-8")
    typer.echo(f"wrote {out}")


@ontology_app.command("import-owl")
def ontology_import_owl(
    path: Path,
    allow_text_change: bool = typer.Option(
        False, help="Permit rewording existing classes; off so structure cannot alter them."
    ),
) -> None:
    """Create or merge an ontology from OWL (Turtle), relations included."""
    from hontology.ontology import owl, snapshots

    with session_scope() as session:
        ontology = owl.import_turtle(
            session, path.read_text(encoding="utf-8"), allow_text_change=allow_text_change
        )
        ref = snapshots.resolve_current(session, ontology.id)
    typer.echo(
        f"imported {ontology.slug!r} (id {ontology.id}) as {ref.version}"
        f"{' (new version)' if ref.created else ''}"
    )


@ontology_app.command("links-import")
def ontology_links_import(ontology_id: int, path: Path) -> None:
    """Apply hand-curated concept↔code links from a CSV (concept,system,code)."""
    import csv

    from hontology.retrieve import similarity

    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    with session_scope() as session:
        result = similarity.import_links(session, ontology_id, rows)
    typer.echo(f"linked {result['links']} code(s) across {result['concepts']} concept(s)")


@ontology_app.command("lint")
def ontology_lint_command(ontology_id: int) -> None:
    """Health checks: strength drift, near-duplicates, thin definitions."""
    from hontology.ontology import lint as lint_module

    with session_scope() as session:
        result = lint_module.lint(session, ontology_id)

    typer.echo(
        f"{result['concepts']} concept(s): {result['warnings']} warning(s), "
        f"{result['info']} note(s)"
    )
    for finding in result["findings"]:
        colour = typer.colors.YELLOW if finding["severity"] == "warning" else None
        typer.secho(
            f"  [{finding['severity']}] {finding['check']}: {finding['concept_name']}",
            fg=colour,
        )
        typer.echo(f"      {finding['message']}")


@ontology_app.command("version")
def ontology_version(ontology_id: int) -> None:
    """Resolve the current version, minting one only if the wording changed."""
    with session_scope() as session:
        ref = snapshots.resolve_current(session, ontology_id)
    status = "minted" if ref.created else "unchanged"
    typer.echo(f"{ref.version}  ({status}, {ref.n_concepts} concepts)")


# ---------------------------------------------------------------------------
# Ingest
#
# One-shot commands are the baseline. `watch` is the only continuous one, and it
# has to be started deliberately — nothing else in the system spawns it.
# ---------------------------------------------------------------------------


@ingest_app.command("status")
def ingest_status() -> None:
    """Show the watermark, how far behind it is, and where the gaps are."""
    with session_scope() as session:
        info = service.status(session)

    lag = info["lag_slices"]
    typer.echo(f"watermark    {info['watermark'] or '(never ingested)'}")
    if lag is None:
        typer.echo("lag          n/a — nothing ingested yet")
    else:
        typer.secho(
            f"lag          {lag} slice(s) ≈ {lag * 15} min",
            fg=typer.colors.GREEN if lag <= 2 else typer.colors.YELLOW,
        )
    typer.echo(f"slices       {info['slice_counts'] or '(none)'}")
    typer.echo(
        f"documents    {info['documents']} ({info['documents_unfetched']} not yet fetched)"
    )


@ingest_app.command("once")
def ingest_once(
    max_slices: int = typer.Option(32, help="Upper bound on slices for this pass."),
) -> None:
    """Catch up to the newest published slice, then exit."""
    with session_scope() as session:
        result = service.catch_up(session, max_slices=max_slices)
    typer.echo(
        f"latest={result['latest_published']} watermark={result['watermark']} "
        f"processed={result['processed']} remaining={result['remaining']}"
    )
    for row in result["slices"]:
        typer.echo(f"  {row['slice_key']}  {row['status']:<8} {row.get('rows', '')}")


@ingest_app.command("backfill")
def ingest_backfill(start: str, end: str) -> None:
    """Ingest an explicit window. Does not move the watermark."""
    with session_scope() as session:
        result = service.backfill(session, start, end)
    typer.echo(f"processed {result['processed']} slice(s) from {start} to {end}")


@ingest_app.command("themes")
def ingest_themes() -> None:
    """Load the GKG theme vocabulary as a code system, for theme links."""
    from hontology.ingest import themes

    with session_scope() as session:
        result = themes.ingest(session)
    typer.echo(f"gkg themes: {result['inserted']} new, {result['total']} total")


@ingest_app.command("calendar")
def ingest_calendar(
    calendar_path: Path,
    before: int = typer.Option(1, help="Days of feed before each entry's date."),
    after: int = typer.Option(2, help="Days of feed after each entry's date."),
) -> None:
    """Backfill both feeds for every window an event calendar needs.

    The event export is kept whole; the knowledge graph, roughly seventy times
    larger, keeps only articles mentioning a country the calendar names on that
    day. Each slice commits on its own, so an interrupted backfill resumes where
    it stopped.
    """
    from hontology.evalkit import calendar

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(message)s")
    entries = calendar.load(calendar_path)
    with session_scope() as session:
        loci = calendar.loci_for(session, entries)
    days = calendar.days_to_ingest(entries, loci, before=before, after=after)
    typer.echo(f"{len(days)} day(s), {len(days) * 96} slice(s) per feed")

    totals: dict[str, int] = {}
    for index, (day, places) in enumerate(days.items(), start=1):
        start = day.strftime("%Y%m%d") + "000000"
        for key in [start, *gdelt.keys_between(start, day.strftime("%Y%m%d") + "234500")]:
            for feed, scope in ((service.FEED, None), (service.FEED_GKG, places)):
                with session_scope() as session:
                    result = service.ingest_slice(session, key, feed=feed, loci=scope)
                status_key = "skipped" if result.get("skipped") else result["status"]
                totals[status_key] = totals.get(status_key, 0) + 1
        typer.echo(f"[{index}/{len(days)}] {day}  {totals}")


@ingest_app.command("repair-export-loci")
def ingest_repair_export_loci(
    workers: int = typer.Option(8, help="Slices downloaded in parallel."),
    limit: int | None = typer.Option(None, help="Repair at most this many slices."),
) -> None:
    """Give old event rows their country, by re-reading their export slices.

    Events ingested before the column fix were stored without a country. Each
    slice is downloaded again, read with the corrected parser and its rows
    updated in place, committing per slice, so the repair can stop and resume.
    """
    from concurrent.futures import ThreadPoolExecutor

    from hontology.ingest.loci import by_fips

    settings = get_settings()
    with session_scope() as session:
        keys = service.slices_missing_export_loci(session)
        lookup = by_fips(session)
    keys = keys[:limit] if limit is not None else keys
    typer.echo(f"{len(keys)} slice(s) to repair")

    def fetch(key: str) -> tuple[str, dict[str, int] | None]:
        try:
            payload = gdelt.fetch_slice(settings.gdelt_base_url, key)
        except Exception:  # noqa: BLE001 - a failed slice is reported, not fatal
            return key, None
        return key, service.export_loci(payload, lookup)

    updated = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (key, loci) in enumerate(pool.map(fetch, keys), start=1):
            if loci is None:
                failed += 1
            else:
                with session_scope() as session:
                    updated += service.apply_export_loci(session, key, loci)
            if done % 500 == 0 or done == len(keys):
                typer.echo(f"[{done}/{len(keys)}] {updated} event(s) located, {failed} failed")


@ingest_app.command("calendar-preview")
def ingest_calendar_preview(
    calendar_path: Path,
    ontology_id: int | None = typer.Option(None, help="Also count what the filter keeps."),
    before: int = typer.Option(1),
    after: int = typer.Option(2),
) -> None:
    """How many documents each calendar window holds, before fetching any."""
    from hontology.db.models import Document
    from hontology.evalkit import calendar
    from hontology.ingest import filter as ingest_filter

    entries = calendar.load(calendar_path)
    with session_scope() as session:
        loci = calendar.loci_for(session, entries)
        passed = (
            set(ingest_filter.matching_documents(session, ontology_id)) if ontology_id else None
        )
        everything: set[int] = set()
        kept: set[int] = set()
        typer.echo(f"{'entry':40} {'in feed':>8} {'filter':>8}")
        for entry in entries:
            docs = set(
                calendar.window_documents(session, entry, loci, before=before, after=after)
            )
            everything |= docs
            hit = docs & passed if passed is not None else docs
            kept |= hit
            shown = len(hit) if passed is not None else "-"
            typer.echo(f"{entry.id:40} {len(docs):>8} {shown:>8}")
        unfetched = len(
            list(
                session.scalars(
                    select(Document.id).where(
                        Document.id.in_(kept), Document.fetched_at.is_(None)
                    )
                )
            )
        )
    typer.echo(
        f"distinct documents: {len(everything)} in windows, {len(kept)} kept, "
        f"{unfetched} still to fetch"
    )


@ingest_app.command("scrape")
def ingest_scrape(
    limit: int | None = typer.Option(None, help="Max documents this run (default: budget)."),
    retry_failed: bool = typer.Option(
        False, help="Also re-attempt URLs already recorded as failures."
    ),
    reader_proxy: bool = typer.Option(
        False, help="Allow the third-party reader service as a last resort."
    ),
    ontology_id: int | None = typer.Option(
        None,
        help="Only fetch documents whose feed codes reach this ontology's concepts. "
        "Off by default; meaningless for an ontology with no code links.",
    ),
    calendar_path: Path | None = typer.Option(
        None, "--calendar", help="Only fetch documents inside this calendar's windows."
    ),
    before: int = typer.Option(1, help="Calendar window: days before each date."),
    after: int = typer.Option(2, help="Calendar window: days after each date."),
) -> None:
    """Fetch article text for documents that do not have it yet.

    Work commits in batches, so a long scrape that is stopped or crashes keeps
    everything but its last batch instead of losing the whole run.
    """
    from hontology.ingest import filter as ingest_filter

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(message)s")
    with session_scope() as session:
        document_ids = None
        if calendar_path is not None:
            from hontology.evalkit import calendar

            document_ids = calendar.all_window_documents(
                session, calendar.load(calendar_path), before=before, after=after
            )
        if ontology_id is not None:
            # Resolved once here rather than per batch: the answer does not
            # change while fetching. With a calendar, only its windows are
            # matched; matching every feed record loads all of them.
            matches = set(
                ingest_filter.matching_documents(
                    session,
                    ontology_id,
                    document_ids=sorted(document_ids) if document_ids is not None else None,
                )
            )
            if not matches:
                typer.secho(
                    "no documents match this ontology's code links; nothing to fetch",
                    fg=typer.colors.YELLOW,
                )
                raise typer.Exit(1)
            document_ids = matches if document_ids is None else document_ids & matches
    allowed = sorted(document_ids) if document_ids is not None else None
    if allowed is not None:
        typer.echo(f"{len(allowed)} document(s) in scope")

    budget = limit if limit is not None else get_settings().scrape_budget

    def scrape_batch(held: list[int]) -> dict:
        with session_scope() as session:
            return scrape.scrape_pending(
                session,
                limit=min(SCRAPE_BATCH, budget),
                retry_failed=retry_failed,
                use_reader_proxy=reader_proxy,
                document_ids=allowed,
                exclude=held,
            )

    def report(totals: dict[str, int]) -> None:
        typer.echo(
            f"attempted {totals['attempted']}: {totals['ok']} ok, {totals['junk']} junk, "
            f"{totals['failed']} failed, {totals['blocked_by_robots']} blocked by robots, "
            f"{totals['deferred']} deferred by crawl delay, "
            f"{totals['retry_later']} left for a later batch after a connection failure"
        )

    if retry_failed:
        # A retry pass would re-attempt the same failures every batch; one is enough.
        first = scrape_batch([])
        totals = {key: first.get(key, 0) for key in scrape.TOTAL_KEYS}
        if totals["attempted"]:
            report(totals)
    else:
        totals = scrape.drain(scrape_batch, budget=budget, on_batch=report)
    if not totals["attempted"]:
        typer.echo("nothing to fetch")


@ingest_app.command("dedup")
def ingest_dedup(
    calendar_path: Path | None = typer.Option(
        None, "--calendar", help="Only documents inside this calendar's windows."
    ),
    before: int = typer.Option(1),
    after: int = typer.Option(2),
    threshold: float = typer.Option(0.8, help="Estimated Jaccard similarity to group at."),
) -> None:
    """Group near-duplicate articles so only one of each is retrieved and judged."""
    from hontology.db.models import Document
    from hontology.ingest import dedup

    with session_scope() as session:
        if calendar_path is not None:
            from hontology.evalkit import calendar

            ids = calendar.all_window_documents(
                session, calendar.load(calendar_path), before=before, after=after
            )
        else:
            ids = set(
                session.scalars(select(Document.id).where(Document.body_path.is_not(None)))
            )
        result = dedup.deduplicate(session, sorted(ids), threshold=threshold)
    copies = result["documents"] - result["representatives"]
    typer.echo(
        f"{result['documents']} fetched document(s): {result['representatives']} to judge, "
        f"{result['newly_marked']} newly marked as copies "
        f"({result['too_short']} too short to fingerprint)"
    )
    if result["documents"]:
        typer.echo(f"judging load cut by {copies / result['documents']:.0%}")


@ingest_app.command("filter-preview")
def ingest_filter_preview(ontology_id: int) -> None:
    """Show what the pre-scrape code filter would keep, without fetching."""
    from hontology.ingest import filter as ingest_filter

    with session_scope() as session:
        report = ingest_filter.preview(session, ontology_id)

    if not report["usable"]:
        typer.secho(report["reason"], fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    typer.echo(f"linked codes        {report['linked_codes']}")
    typer.echo(f"documents total     {report['documents_total']}")
    typer.echo(f"  matching          {report['documents_matching']}")
    typer.echo(f"unfetched           {report['documents_unfetched']}")
    typer.echo(f"  would fetch       {report['unfetched_matching']}")
    typer.echo(f"  would skip        {report['unfetched_skipped']}")
    share = report["share_kept"]
    if share is not None:
        typer.secho(f"share kept          {share:.1%}", fg=typer.colors.GREEN)


@ingest_app.command("watch")
def ingest_watch(
    grace_seconds: float = typer.Option(
        90.0, help="Delay after each quarter-hour boundary before polling."
    ),
    max_slices: int = typer.Option(32, help="Upper bound on slices per pass."),
) -> None:
    """Run continuously, following the feed's 15-minute cadence.

    Opt-in: this is the only command that keeps running, and nothing starts it
    automatically. Stop it with Ctrl-C or SIGTERM; the slice in flight finishes
    and commits first.
    """
    from hontology.ingest.scheduler import Watcher

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    )
    watcher = Watcher(grace_seconds=grace_seconds, max_slices=max_slices)
    watcher.install_signal_handlers()
    watcher.run()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _sigterm_as_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


@run_app.callback()
def run_signals() -> None:
    """Configure and execute detection runs."""
    # A run stopped with SIGTERM (`kill`, a process manager) would otherwise end
    # without raising anything, skipping the handlers that record an interrupted
    # run on its row, and leave it "running" forever. Raised as Ctrl-C is, it is
    # recorded the same way.
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)


def _record_failure(run, exc: BaseException) -> None:
    from datetime import UTC, datetime

    run.status = "failed"
    run.error = (
        "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)[:1000]
    ) or type(exc).__name__
    run.finished_at = datetime.now(UTC)


def _load_run(session, run_id: int):
    """Fetch a run or exit with a clear message rather than a traceback."""
    from hontology.db.models import Run

    run = session.get(Run, run_id)
    if run is None:
        typer.secho(f"no run with id {run_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    return run


@run_app.command("prompts")
def run_prompts() -> None:
    """List the registered prompt templates."""
    from hontology.judge import prompts

    for prompt_id in prompts.available():
        template = prompts.get(prompt_id)
        typer.echo(f"{prompt_id:<20} {template.mode}")


@run_app.command("keys")
def run_keys(config_path: Path, ontology_version: str = "v1") -> None:
    """Show the stage keys a config resolves to, without executing anything.

    Useful for checking what an edit will recompute before paying for it.
    """
    from hontology.evalkit import config as run_config

    normalized = run_config.normalize(json.loads(config_path.read_text(encoding="utf-8")))
    keys = run_config.stage_keys(normalized, ontology_version)
    typer.echo(f"candidates  {keys['candidates']}")
    typer.echo(f"judge       {keys['judge']}")


@run_app.command("list")
def run_list() -> None:
    """List runs with their status and stage keys."""
    from hontology.db.models import Run

    with session_scope() as session:
        runs = list(session.scalars(select(Run).order_by(Run.id)))
        if not runs:
            typer.echo("(no runs yet)")
            return
        for run in runs:
            typer.echo(
                f"{run.id:>4}  {run.status:<10} {run.name:<24} "
                f"{run.ontology_version:<5} cand={run.candidates_key} "
                f"judge={run.judge_key.split('_')[0]}"
            )


@run_app.command("start")
def run_start(
    ontology_id: int,
    config_path: Path,
    name: str | None = typer.Option(None, help="Override the config's name."),
    documents: int = typer.Option(100, help="How many documents to consider."),
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged this pass."),
    skip_judge: bool = typer.Option(False, help="Build candidates only."),
    refresh_embeddings: bool = typer.Option(
        False,
        help="Recompute document embeddings instead of reusing cached ones. For "
        "when the cache itself is the suspect; results are unaffected.",
    ),
    calendar_path: Path | None = typer.Option(
        None,
        "--calendar",
        help="Consider only fetched documents inside this calendar's windows, "
        "instead of the newest --documents.",
    ),
    before: int = typer.Option(1, help="Calendar window: days before each date."),
    after: int = typer.Option(2, help="Calendar window: days after each date."),
) -> None:
    """Execute a run from a JSON config."""
    from hontology.evalkit import runner

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    with session_scope() as session:
        run = runner.create_run(session, ontology_id=ontology_id, config=payload, name=name)
        run_id = run.id
        typer.echo(f"run {run_id}: candidates={run.candidates_key} judge={run.judge_key}")

    document_ids = None
    if calendar_path is not None:
        from hontology.evalkit import calendar

        with session_scope() as session:
            document_ids = sorted(
                calendar.all_window_documents(
                    session, calendar.load(calendar_path), before=before, after=after
                )
            )
            # Which documents a run considered is provenance, not behaviour: it
            # is recorded, never hashed, like the --documents limit.
            run = _load_run(session, run_id)
            run.manifest = (run.manifest or {}) | {
                "documents": {
                    "calendar": str(calendar_path),
                    "window_days": [before, after],
                    "in_windows": len(document_ids),
                }
            }
        typer.echo(f"calendar: {len(document_ids)} document(s) in its windows")

    with session_scope() as session:
        result = runner.execute(
            session,
            _load_run(session, run_id),
            document_limit=documents if document_ids is None else len(document_ids),
            judge_limit=judge_limit,
            skip_judge=skip_judge,
            refresh_embeddings=refresh_embeddings,
            document_ids=document_ids,
        )

    typer.echo(f"candidates  {result['candidates']}")
    if result.get("reused_from_run"):
        typer.secho(
            f"            retrieval reused from run {result['reused_from_run']}",
            fg=typer.colors.GREEN,
        )
    if result.get("judge"):
        typer.echo(f"judge       {result['judge']}")
        live = result["liveness"]
        typer.secho(
            f"liveness    {live['clean']}/{live['verdicts']} clean, {live['errors']} errors",
            fg=typer.colors.GREEN if live["ok"] else typer.colors.RED,
        )


@run_app.command("sample")
def run_sample(
    ontology_id: int,
    config_path: Path,
    manifest_path: Path,
    candidates_from: int = typer.Option(
        ..., help="Reuse this run's retrieval for the sample's documents."
    ),
    first: int | None = typer.Option(
        None, help="Judge only the first N documents in the sample's frozen order."
    ),
    run_id: int | None = typer.Option(None, help="Extend this run instead of creating one."),
) -> None:
    """Judge a labelled sample's documents with another run's retrieval.

    An arm can be compared at article level as soon as documents are labelled,
    without judging every calendar window first. Rerun with a larger --first
    and the same --run-id as labelling continues; only new documents are judged.
    The run's calendar entries stay unprocessed until `run calendar` extends it.
    """
    from datetime import UTC, datetime

    from hontology.evalkit import calendar_run, runner

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(message)s")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = [row["document_id"] for row in manifest["order"]]
    document_ids = order[:first] if first is not None else order

    with session_scope() as session:
        if run_id is None:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            run = runner.create_run(session, ontology_id=ontology_id, config=payload)
        else:
            run = _load_run(session, run_id)
        # Same leaves, or the arms would be answering different questions.
        calendar_run.check_same_leaves(
            session, _load_run(session, candidates_from), run.ontology_id
        )
        run.status = "running"
        run.started_at = run.started_at or datetime.now(UTC)
        # A resumed run must not keep the finish time or error of its last pass.
        run.finished_at = None
        run.error = None
        run.manifest = (
            {"calendar_done": []}
            | (run.manifest or {})
            | {
                "sample": {
                    "manifest": str(manifest_path),
                    "order_sha256": manifest["order_sha256"],
                    "judged_first": len(document_ids),
                    "candidates_from": candidates_from,
                }
            }
        )
        session.commit()
        typer.echo(f"run {run.id}: judging {len(document_ids)} sample document(s)")
        try:
            result = calendar_run.judge_documents(
                session, run, source_run_id=candidates_from, document_ids=document_ids
            )
        except BaseException as exc:
            # Verdicts already committed per pair are kept, for the next resume.
            session.rollback()
            _record_failure(run, exc)
            session.commit()
            raise
        run.status = "done"
        run.finished_at = datetime.now(UTC)
        judge = result["judge"] or {}
        typer.echo(
            f"run {run.id}: {result['retrieved']} of {result['documents']} document(s) "
            f"retrieved by run {candidates_from}; judged {judge.get('judged', 0)}, "
            f"matched {judge.get('matched', 0)}, {judge.get('calls', 0)} call(s)"
        )


@run_app.command("calendar")
def run_calendar(
    ontology_id: int,
    config_path: Path,
    calendar_path: Path,
    run_id: int | None = typer.Option(None, help="Resume this run instead of creating one."),
    budget: int | None = typer.Option(
        None, help="Pairs judged per window, highest retrieval score first. Default: all."
    ),
    before: int = typer.Option(1, help="Calendar window: days before each date."),
    after: int = typer.Option(2, help="Calendar window: days after each date."),
    poll: int = typer.Option(120, help="Seconds to wait when no window is ready yet."),
    prepare_only: bool = typer.Option(
        False,
        help="Scrape, deduplicate and retrieve, but judge nothing and mark nothing "
        "finished, to see each window's judging volume first.",
    ),
    candidates_from: int | None = typer.Option(
        None,
        help="Reuse this run's retrieval: judge exactly its documents in each window, "
        "once it has finished that window. For comparing judging arms.",
    ),
) -> None:
    """Run a calendar window by window, as soon as each window's feed is ingested.

    Each ready window is scraped, deduplicated, retrieved and judged into one
    run; finished entries are recorded on the run, so a restart with --run-id
    continues where it stopped. Score it afterwards with `eval calendar`.
    """
    import time
    from datetime import UTC, datetime

    from hontology.evalkit import calendar, calendar_run, runner

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(message)s")
    entries = calendar.load(calendar_path)

    with session_scope() as session:
        loci = calendar.loci_for(session, entries)
        if run_id is None:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            run = runner.create_run(session, ontology_id=ontology_id, config=payload)
        else:
            run = _load_run(session, run_id)
        run.status = "running"
        run.stage = "calendar"
        run.started_at = run.started_at or datetime.now(UTC)
        run.finished_at = None
        run.error = None
        run.manifest = (run.manifest or {}) | {
            "documents": {
                "calendar": str(calendar_path),
                "window_days": [before, after],
                "budget_per_window": budget,
                "candidates_from": candidates_from,
            }
        }
        if candidates_from is not None:
            # Same leaves, or the arms would be answering different questions.
            calendar_run.check_same_leaves(
                session, _load_run(session, candidates_from), run.ontology_id
            )
        run_id = run.id
        # A prepare-only pass tracks its own progress, so a later judging pass
        # still visits every window.
        progress_key = "calendar_prepared" if prepare_only else "calendar_done"
        finished = set((run.manifest or {}).get(progress_key, []))
        # Windows the source published nothing for: set aside, never scored.
        unobservable = set((run.manifest or {}).get("calendar_unobservable", []))
    typer.echo(f"run {run_id}: {len(finished)}/{len(entries)} entries already done")
    unobservable &= {e.id for e in entries}
    if unobservable:
        typer.echo(f"  set aside, no source data: {', '.join(sorted(unobservable))}")

    try:
        while len(finished | unobservable) < len(entries):
            progressed = False
            for entry in entries:
                if entry.id in finished or entry.id in unobservable:
                    continue
                places = [loci[c] for c in entry.countries]
                with session_scope() as session:
                    ready, failed = calendar_run.window_status(
                        session, entry, places, before=before, after=after
                    )
                if not ready and failed:
                    for feed, key in failed:
                        with session_scope() as session:
                            scope = places if feed == service.FEED_GKG else None
                            service.ingest_slice(session, key, feed=feed, loci=scope)
                    with session_scope() as session:
                        ready, _ = calendar_run.window_status(
                            session, entry, places, before=before, after=after
                        )
                if ready:
                    with session_scope() as session:
                        if calendar_run.window_unobservable(
                            session, entry, places, before=before, after=after
                        ):
                            unobservable.add(entry.id)
                            run = _load_run(session, run_id)
                            stored = set((run.manifest or {}).get("calendar_unobservable", []))
                            run.manifest = (run.manifest or {}) | {
                                "calendar_unobservable": sorted(stored | {entry.id})
                            }
                            typer.echo(
                                f"[set aside] {entry.id}: the source published nothing "
                                "for this window; not scored"
                            )
                            progressed = True
                            continue
                if ready and candidates_from is not None:
                    # Reusing another run's retrieval: wait until it has
                    # finished this window, so the documents are all there.
                    with session_scope() as session:
                        source = _load_run(session, candidates_from)
                        ready = entry.id in (source.manifest or {}).get("calendar_done", [])
                if not ready:
                    continue
                with session_scope() as session:
                    run = _load_run(session, run_id)
                    summary = calendar_run.process_entry(
                        session,
                        run,
                        entry,
                        places,
                        before=before,
                        after=after,
                        budget=budget,
                        judge=not prepare_only,
                        candidates_from=candidates_from,
                    )
                    finished.add(entry.id)
                    run.manifest = (run.manifest or {}) | {progress_key: sorted(finished)}
                judge = summary["judge"] or {}
                scope_note = f"{summary['passed_filter']} in scope"
                unique_note = f"{summary['representatives']} unique"
                judge_note = (
                    f"{summary['pairs_selected']} pair(s) selected"
                    if prepare_only
                    else f"judged {judge.get('judged', 0)}, matched {judge.get('matched', 0)}"
                )
                typer.echo(
                    f"[{len(finished)}/{len(entries)}] {entry.id}: "
                    f"{scope_note}, {unique_note}, {judge_note}"
                )
                progressed = True
            if not progressed and len(finished | unobservable) < len(entries):
                time.sleep(poll)
    except BaseException as exc:
        with session_scope() as session:
            _record_failure(_load_run(session, run_id), exc)
        raise

    with session_scope() as session:
        run = _load_run(session, run_id)
        if prepare_only:
            # Prepared, not judged: left resumable rather than marked done.
            run.status = "candidates"
            run.stage = "judge"
            typer.echo(f"run {run_id} prepared; judge it with --run-id {run_id}")
            return
        run.status = "done"
        run.stage = None
        run.finished_at = datetime.now(UTC)
    typer.echo(
        f"run {run_id} done; score it with: hontology eval calendar {run_id} {calendar_path}"
    )


@run_app.command("resume")
def run_resume(
    run_id: int,
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged this pass."),
) -> None:
    """Continue a run, skipping pairs already judged."""
    from hontology.evalkit import runner

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    with session_scope() as session:
        result = runner.execute(session, _load_run(session, run_id), judge_limit=judge_limit)
    typer.echo(f"judge  {result['judge']}")


# ---------------------------------------------------------------------------
# Ground-truth bank portability
# ---------------------------------------------------------------------------


@labels_app.command("export")
def labels_export(
    ontology_id: int,
    out: Path | None = typer.Option(None, help="Write to a file instead of stdout."),
    observations: bool = typer.Option(False, help="Export observations instead of labels."),
) -> None:
    """Export the bank as CSV, keyed by document URL and concept name."""
    from hontology.evalkit import label_io

    with session_scope() as session:
        text_out = (
            label_io.export_observations(session, ontology_id)
            if observations
            else label_io.export_labels(session, ontology_id)
        )

    if out is None:
        typer.echo(text_out)
    else:
        out.write_text(text_out, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(text_out.splitlines()) - 1} row(s))")


@labels_app.command("sample")
def labels_sample(
    run_id: int,
    calendar_path: Path,
    out: Path = typer.Option(..., help="Manifest JSON to write."),
    seed: int = typer.Option(20261006, help="Fixes the order; record it with the sample."),
    allocation: str = typer.Option(
        "equal-per-window",
        help="equal-per-window (each window an equal share, metrics weighted by "
        "window size) or proportional.",
    ),
    before: int = typer.Option(1),
    after: int = typer.Option(2),
) -> None:
    """Draw the labelled sample: every representative in the calendar's windows
    that the run processed, in a frozen order whose every prefix is a stratified
    random sample."""
    import hashlib

    from hontology.evalkit import calendar, sample

    entries = calendar.load(calendar_path)
    digest = hashlib.sha256(calendar_path.read_bytes()).hexdigest()
    with session_scope() as session:
        documents, late = sample.frame(session, run_id, entries, before=before, after=after)
    ordered = sample.frozen_order(documents, seed, allocation)
    record = sample.manifest(
        run_id, digest, seed, ordered, late_arrivals=late, allocation=allocation
    )
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    typer.echo(
        f"{record['size']} document(s) in {len(record['strata'])} strata; "
        f"order {record['order_sha256'][:12]}; "
        f"{late} left out, fetched after the run processed their window"
    )


@labels_app.command("sample-sheet")
def labels_sample_sheet(
    ontology_id: int,
    manifest_path: Path,
    out: Path = typer.Option(..., help="Labelling CSV to write."),
    start: int = typer.Option(1, help="First position in the frozen order (1-based)."),
    count: int = typer.Option(120, help="How many documents."),
) -> None:
    """Write the next documents to label, in frozen order, plus the concept list.

    Fill `concepts` with the names that apply, separated by `;`, or `none`.
    Leave it blank for a document not yet read; blanks are skipped on import.
    """
    import csv

    from hontology.db.models import Concept, Document
    from hontology.evalkit.document_labels import DOCUMENT_LABEL_COLUMNS

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk = record["order"][start - 1 : start - 1 + count]
    cache = get_settings().scrape_cache_dir
    with session_scope() as session:
        documents = {
            d.id: d
            for d in session.scalars(
                select(Document).where(Document.id.in_([r["document_id"] for r in chunk]))
            )
        }
        from hontology.ontology import hierarchy

        leaf_ids = hierarchy.leaves(session, ontology_id)
        concepts = [
            c
            for c in session.scalars(
                select(Concept).where(Concept.ontology_id == ontology_id).order_by(Concept.name)
            )
            if c.id in leaf_ids  # people label leaves; internal classes are derived
        ]
        rows = []
        for position, row in enumerate(chunk, start=start):
            document = documents[row["document_id"]]
            path = cache / (document.body_path or "")
            body = (
                path.read_text(encoding="utf-8") if document.body_path and path.exists() else ""
            )
            rows.append(
                {
                    "position": position,
                    "document_url": document.url,
                    "title": document.title or body.split("\n", 1)[0][:160],
                    "excerpt": " ".join(body.split())[:700],
                    "concepts": "",
                    "note": "",
                }
            )
        reference = "\n\n".join(
            f"{c.name}\n  {c.definition or ''}\n  counts when: {c.inclusion_criteria or '-'}"
            f"\n  not when: {c.exclusion_criteria or '-'}"
            for c in concepts
        )
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCUMENT_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    concept_file = out.with_suffix(".concepts.txt")
    concept_file.write_text(reference + "\n", encoding="utf-8")
    typer.echo(f"{len(rows)} document(s) to {out}; concept reference in {concept_file}")


@labels_app.command("import-documents")
def labels_import_documents(ontology_id: int, path: Path) -> None:
    """Read whole-document labels: listed concepts positive, all others negative."""
    from hontology.evalkit.document_labels import import_document_labels

    with session_scope() as session:
        report = import_document_labels(session, ontology_id, path.read_text(encoding="utf-8"))
    typer.echo(
        f"{report['documents']} document(s): {report['positives']} positive, "
        f"{report['negatives']} negative label(s); "
        f"{report['skipped_blank']} blank row(s) skipped"
    )
    for error in report["errors"]:
        typer.secho(f"  {error}", fg=typer.colors.YELLOW)


@labels_app.command("annotations-import")
def labels_annotations_import(
    ontology_id: int,
    path: Path,
    name: str = typer.Option(..., help="The set's name, e.g. claude-blind-001."),
    description: str | None = typer.Option(
        None, help="Who annotated and how, for whoever reads a score against it."
    ),
) -> None:
    """Store a whole-document labels file as a machine annotation set.

    Kept apart from the human label bank: scores against it measure agreement
    with that annotator. Re-importing under the same name extends the set.
    """
    from hontology.evalkit import annotations

    with session_scope() as session:
        try:
            report = annotations.import_set(
                session,
                ontology_id,
                name,
                path.read_text(encoding="utf-8"),
                description=description,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"{report['set']}: {report['articles']} article(s), {report['pairs']} pair(s), "
        f"{report['positives']} positive; {report['skipped_blank']} blank row(s) skipped"
    )


@labels_app.command("import")
def labels_import(
    ontology_id: int,
    path: Path,
    overwrite: bool = typer.Option(
        False,
        help="Replace labels that already exist. Off by default so an "
        "import cannot silently destroy adjudicated work.",
    ),
    observations: bool = typer.Option(False, help="Import observations instead of labels."),
) -> None:
    """Import a CSV bank, matching on document URL and concept name."""
    from hontology.evalkit import label_io

    csv_text = path.read_text(encoding="utf-8")
    with session_scope() as session:
        report = (
            label_io.import_observations(session, ontology_id, csv_text)
            if observations
            else label_io.import_labels(session, ontology_id, csv_text, overwrite=overwrite)
        )

    typer.echo(
        f"created {report['created']}, updated {report['updated']}, "
        f"skipped {report['skipped_existing']} existing"
    )
    if report.get("documents_created"):
        typer.echo(f"created {report['documents_created']} stub document(s)")
    if report["unknown_concepts"]:
        typer.secho(
            f"skipped unknown concept(s): {', '.join(report['unknown_concepts'])}",
            fg=typer.colors.YELLOW,
        )
    if report["bad_row_count"]:
        typer.secho(f"{report['bad_row_count']} unreadable row(s):", fg=typer.colors.YELLOW)
        for line in report["bad_rows"]:
            typer.echo(f"  {line}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@eval_app.command("run")
def eval_run(
    run_id: int,
    include_machine: bool = typer.Option(
        False, help="Count un-adjudicated machine labels. Off by default."
    ),
    include_stale: bool = typer.Option(
        False, help="Count labels whose concept was reworded after labelling."
    ),
    record: bool = typer.Option(False, help="Also write the result to the leaderboard."),
) -> None:
    """Per-stage metrics for one run, with intervals and denominators."""
    from hontology.evalkit import evaluate, warehouse

    with session_scope() as session:
        evaluation = evaluate.evaluate_run(
            session, run_id, include_machine=include_machine, include_stale=include_stale
        )
        typer.echo(evaluate.format_report(evaluation))
        if record:
            run = _load_run(session, run_id)
            warehouse.record(
                evaluation,
                config=run.config,
                candidates_key=run.candidates_key,
                judge_key=run.judge_key,
            )
            typer.secho("recorded to the leaderboard", fg=typer.colors.GREEN)


@eval_app.command("breakdown")
def eval_breakdown(
    run_id: int,
    dimension: str = typer.Option("concept", help="concept | family | category | locus"),
    include_machine: bool = typer.Option(False, help="Count machine labels."),
) -> None:
    """Metrics sliced, so a systematic failure is visible rather than pooled away."""
    from hontology.evalkit import breakdown

    with session_scope() as session:
        rows = breakdown.breakdown(
            session, run_id, dimension=dimension, include_machine=include_machine
        )
    typer.echo(breakdown.format_breakdown(rows, dimension))


@eval_app.command("errors")
def eval_errors(
    run_id: int,
    kind: str | None = typer.Option(None, help="false_positive | false_negative"),
    include_machine: bool = typer.Option(False, help="Count machine labels."),
    limit: int = typer.Option(20, help="How many to show."),
) -> None:
    """Every misclassified pair, with the model's evidence and reasoning."""
    from hontology.evalkit import errors

    with session_scope() as session:
        rows = errors.triage(
            session, run_id, kind=kind, include_machine=include_machine, limit=limit
        )
        stats = errors.summary(session, run_id, include_machine=include_machine)
    typer.echo(errors.format_triage(rows))
    typer.echo(
        f"{stats['total']} error(s): {stats['false_positives']} FP, "
        f"{stats['false_negatives']} FN"
    )
    for concept, counts in list(stats["by_concept"].items())[:5]:
        typer.echo(
            f"  {concept:<30} FP={counts['false_positive']} FN={counts['false_negative']}"
        )


@eval_app.command("filter-report")
def eval_filter_report(ontology_id: int) -> None:
    """Per-code cost and benefit of the pre-scrape filter mapping."""
    from hontology.evalkit import filter_report

    with session_scope() as session:
        typer.echo(filter_report.format_report(filter_report.report(session, ontology_id)))


@eval_app.command("sweep")
def eval_sweep(
    ontology_id: int,
    sweep_path: Path,
    execute: bool = typer.Option(False, help="Run the plan, not just print it."),
    documents: int = typer.Option(50, help="Documents per cell."),
    judge_limit: int | None = typer.Option(None, help="Cap pairs judged per cell."),
) -> None:
    """Plan or run a parameter sweep. Cells already run are skipped."""
    from hontology.evalkit import sweep as sweep_module

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    spec = json.loads(sweep_path.read_text(encoding="utf-8"))

    with session_scope() as session:
        plan = sweep_module.plan(
            session,
            ontology_id,
            base=spec.get("base", {}),
            axes=spec.get("axes", {}),
            name_prefix=spec.get("name", "sweep"),
        )
        summary = plan.as_dict()

    typer.echo(
        f"{summary['total']} cell(s): {summary['done']} already run, "
        f"{summary['todo']} to do, {summary['distinct_candidate_keys']} distinct "
        f"retrieval key(s)"
    )
    for cell in summary["cells"]:
        mark = "done" if cell["done"] else "todo"
        typer.echo(f"  [{mark}] {cell['name']}  cand={cell['candidates_key']}")

    if not execute:
        typer.secho("\nplan only — pass --execute to run", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    with session_scope() as session:
        plan = sweep_module.plan(
            session,
            ontology_id,
            base=spec.get("base", {}),
            axes=spec.get("axes", {}),
            name_prefix=spec.get("name", "sweep"),
        )
        results = sweep_module.execute(
            session, ontology_id, plan, document_limit=documents, judge_limit=judge_limit
        )
    typer.secho(f"\nran {len(results)} cell(s)", fg=typer.colors.GREEN)


@eval_app.command("funnel")
def eval_funnel(run_id: int) -> None:
    """Where the volume went, stage by stage. Needs no labels."""
    from hontology.evalkit import funnel

    with session_scope() as session:
        typer.echo(funnel.format_funnel(funnel.funnel(session, run_id)))


@eval_app.command("calendar")
def eval_calendar(
    run_id: int,
    calendar_path: Path,
    before: int = typer.Option(1, help="Days of feed before each entry's date."),
    after: int = typer.Option(2, help="Days of feed after each entry's date."),
    out: Path | None = typer.Option(None, help="Also write the full result as JSON."),
) -> None:
    """Event-level results against a calendar of known events. Needs no labels."""
    from hontology.evalkit import calendar, calendar_score
    from hontology.evalkit.metrics import format_ci

    with session_scope() as session:
        result = calendar_score.evaluate(
            session, run_id, calendar.load(calendar_path), before=before, after=after
        )
    if out is not None:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # `unique` sits after `fetched`: what is left once near-duplicates collapse.
    columns = (*calendar.STAGES[:3], "unique", *calendar.STAGES[3:])
    typer.echo(
        f"{'entry':34} {'kind':9} " + " ".join(f"{c[:8]:>8}" for c in columns) + "  result"
    )
    for row in result["entries"]:
        verdict = (
            ("FALSE ALARM" if row["kind"] == calendar.CONTROL else "detected")
            if row["detected"]
            else ("quiet" if row["kind"] == calendar.CONTROL else f"lost at {row['lost_at']}")
        )
        if row["detected"]:
            # A person's review of the matches: confirmed, rejected, or not yet.
            verdict += {True: " (confirmed)", False: " (rejected)", None: " (unreviewed)"}[
                row["verified"]
            ]
            if row["kind"] == calendar.CONTROL and row["verified"] is True:
                verdict = "control withdrawn: a real instance was found"
        typer.echo(
            f"{row['id'][:34]:34} {row['kind']:9} "
            + " ".join(f"{row[c]:>8}" for c in columns)
            + f"  {verdict}"
        )

    typer.echo("")
    for label, key in (
        ("event recall", "event_recall"),
        ("precursor recall", "precursor_recall"),
        ("false alarm rate", "false_alarm_rate"),
    ):
        stat = result["summary"][key]
        rate = "-" if stat["rate"] is None else f"{stat['rate']:.2f}"
        typer.echo(f"{label:18} {stat['hits']}/{stat['n']}  {rate}  {format_ci(*stat['ci'])}")
    typer.echo(f"positives lost at  {result['summary']['positives_lost_at']}")
    verified = result["summary"]["verified"]
    typer.echo("verified by review:")
    for label, key in (
        ("  event recall", "event_recall"),
        ("  precursor recall", "precursor_recall"),
        ("  false alarm rate", "false_alarm_rate"),
    ):
        stat = verified[key]
        rate = "-" if stat["rate"] is None else f"{stat['rate']:.2f}"
        typer.echo(f"{label:18} {stat['hits']}/{stat['n']}  {rate}  {format_ci(*stat['ci'])}")
    if verified["controls_withdrawn"]:
        typer.echo(f"  withdrawn        {', '.join(verified['controls_withdrawn'])}")
    if verified["pending_review"]:
        pending = verified["pending_review"]
        typer.echo(f"  pending review   {len(pending)}: {', '.join(pending)}")
    cost = result["cost"]
    typer.echo(
        f"judging cost       {cost['pairs']} pair(s), {cost['input_tokens']} in / "
        f"{cost['output_tokens']} out tokens, {cost['seconds'] / 3600:.1f} h"
    )
    if result["not_processed"]:
        skipped = result["not_processed"]
        typer.echo(f"not processed      {len(skipped)}, not scored: {', '.join(skipped)}")
    for lead in result["lead_times"]:
        typer.echo(
            f"lead  {lead['precursor']} -> {lead['disruption']}: {lead['lead_days']} day(s)"
            f"{'' if lead['disruption_detected'] else '  (disruption itself missed)'}"
        )


REVIEW_COLUMNS = (
    "entry_id",
    "kind",
    "concept",
    "countries",
    "date",
    "description",
    "first_seen",
    "url",
    "evidence",
    "confirmed",
    "note",
)


@eval_app.command("arms")
def eval_arms(
    baseline: int,
    calendar_path: Path,
    arm: list[int] = typer.Option([], "--arm", help="An arm's run id; repeat for several."),
    manifest_path: Path | None = typer.Option(
        None, "--manifest", help="Labelled-sample manifest."
    ),
    out: Path | None = typer.Option(None, help="Write the full report as JSON."),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Write the report as Markdown tables, for the write-up."
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        help="Score against this whole-document labels CSV instead of the label bank.",
    ),
    annotator: str | None = typer.Option(
        None,
        "--annotator",
        help="Score against this machine annotation set instead of the label bank.",
    ),
    before: int = typer.Option(1),
    after: int = typer.Option(2),
) -> None:
    """The pre-registered comparison: every arm against the baseline, both levels."""
    from hontology.db.models import Run
    from hontology.evalkit import annotations, arms, arms_report, calendar
    from hontology.evalkit.document_labels import document_label_map

    if labels_path is not None and annotator is not None:
        raise typer.BadParameter("give --labels or --annotator, not both")
    with session_scope() as session:
        labels = None
        if labels_path is not None or annotator is not None:
            baseline_run = session.get(Run, baseline)
            if baseline_run is None:
                raise typer.BadParameter(f"run {baseline} does not exist")
            if labels_path is not None:
                labels = document_label_map(
                    session, baseline_run.ontology_id, labels_path.read_text(encoding="utf-8")
                )
            else:
                try:
                    labels = annotations.truth(session, baseline_run.ontology_id, annotator)
                except LookupError as exc:
                    raise typer.BadParameter(str(exc)) from exc
        report = arms.compare_arms(
            session,
            baseline,
            arm,
            calendar.load(calendar_path),
            arms.load_manifest(manifest_path),
            before=before,
            after=after,
            labels=labels,
        )
    if out is not None:
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if markdown is not None:
        markdown.write_text(arms_report.render_markdown(report), encoding="utf-8")

    def rate(stat: dict) -> str:
        value = "-" if stat["rate"] is None else f"{stat['rate']:.2f}"
        return f"{stat['hits']}/{stat['n']} {value}"

    for run_id, run in report["runs"].items():
        verified = run["calendar"]["verified"]
        cost = run["cost"]
        typer.echo(f"run {run_id}{' (baseline)' if run_id == baseline else ''}")
        typer.echo(
            f"  events    raw {rate(run['calendar']['event_recall'])}   "
            f"verified {rate(verified['event_recall'])}   "
            f"false alarms {rate(verified['false_alarm_rate'])}"
        )
        if "article" in run:
            scores = run["article"]["end_to_end"]
            typer.echo(
                f"  articles  P {scores['precision'] or 0:.3f} {scores['precision_ci']}  "
                f"R {scores['recall'] or 0:.3f} {scores['recall_ci']}  "
                f"F1 {scores['f1'] or 0:.3f} {scores['f1_ci']}"
            )
        typer.echo(
            f"  cost      {cost['pairs']} verdicts, "
            f"{cost['input_tokens'] + cost['output_tokens']} tokens, "
            f"{cost['seconds'] / 3600:.1f} h"
        )
    sample = report.get("sample")
    if sample:
        typer.echo(f"sample    {sample['labelled_prefix']} labelled in order")
        if sample["out_of_turn"]:
            typer.secho(
                f"          {len(sample['out_of_turn'])} labelled out of turn (not counted)",
                fg=typer.colors.YELLOW,
            )
        if "status" in sample:
            status = sample["status"]
            typer.echo(
                f"stopping  half-widths {status['half_widths']} vs ±{status['target']}: "
                f"{'target met' if status['target_met'] else 'keep labelling'}"
            )
    for arm_id, comparison in report["comparisons"].items():
        f1 = comparison["f1"]
        typer.echo(
            f"arm {arm_id} vs baseline: F1 difference {f1['difference']} {f1['difference_ci']}"
            f" -> {'improvement' if f1['improvement'] else 'not shown'}"
        )


@eval_app.command("calendar-review-export")
def eval_calendar_review_export(
    run_id: int,
    calendar_path: Path,
    out: Path = typer.Option(..., help="CSV to write; fill `confirmed` with yes or no."),
    per_entry: int = typer.Option(5, help="Unreviewed matches listed per entry."),
    before: int = typer.Option(1),
    after: int = typer.Option(2),
) -> None:
    """Write the matches still awaiting review, earliest first, for a person to mark.

    For an event or precursor, `confirmed` asks: does this article describe this
    very event? For a control: does it report a real instance of the concept
    there and then? Entries already confirmed are left out.
    """
    import csv

    from hontology.evalkit import calendar, calendar_score

    entries = calendar.load(calendar_path)
    by_id = {e.id: e for e in entries}
    with session_scope() as session:
        result = calendar_score.evaluate(session, run_id, entries, before=before, after=after)
    rows = []
    for row in result["entries"]:
        if row["verified"] is not None:
            continue
        unreviewed = [m for m in row["matches"] if m["confirmed"] is None][:per_entry]
        for match in unreviewed:
            rows.append(
                {
                    "entry_id": row["id"],
                    "kind": row["kind"],
                    "concept": row["concept"],
                    "countries": row["countries"],
                    "date": row["date"],
                    "description": by_id[row["id"]].description,
                    "first_seen": match["first_seen"],
                    "url": match["url"],
                    "evidence": (match["evidence"] or "").replace("\n", " "),
                    "confirmed": "",
                    "note": "",
                }
            )
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(
        f"{len(rows)} match(es) to review across {len({r['entry_id'] for r in rows})} entries"
    )


@eval_app.command("calendar-review-import")
def eval_calendar_review_import(path: Path) -> None:
    """Read reviewed matches back; rows left blank in `confirmed` are skipped."""
    import csv

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from hontology.db.models import CalendarReview

    answers = {"yes": True, "y": True, "true": True, "1": True}
    answers |= {"no": False, "n": False, "false": False, "0": False}
    saved = skipped = 0
    with path.open(encoding="utf-8", newline="") as handle, session_scope() as session:
        for row in csv.DictReader(handle):
            answer = answers.get((row.get("confirmed") or "").strip().lower())
            if answer is None:
                skipped += 1
                continue
            statement = pg_insert(CalendarReview).values(
                entry_id=row["entry_id"].strip(),
                document_url=row["url"].strip(),
                confirmed=answer,
                note=(row.get("note") or "").strip() or None,
            )
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=["entry_id", "document_url"],
                    set_={"confirmed": answer, "note": statement.excluded.note},
                )
            )
            saved += 1
    typer.echo(f"saved {saved} review(s); {skipped} row(s) left blank")


@eval_app.command("detections")
def eval_detections(
    run_id: int,
    out: Path | None = typer.Option(None, help="Write CSV to a file instead of stdout."),
    events: bool = typer.Option(False, help="Aggregate to one row per (concept, locus, date)."),
    min_confidence: float | None = typer.Option(None, help="Drop weaker detections."),
    verified_only: bool = typer.Option(False, help="Only detections a human has confirmed."),
) -> None:
    """Export what the pipeline found. Every row carries its verification status."""
    from hontology.evalkit import detections as detections_module

    with session_scope() as session:
        exporter = (
            detections_module.export_events if events else detections_module.export_detections
        )
        text_out = exporter(
            session, run_id, min_confidence=min_confidence, verified_only=verified_only
        )
        stats = detections_module.summary(session, run_id)

    if out is None:
        typer.echo(text_out)
    else:
        out.write_text(text_out, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(text_out.splitlines()) - 1} row(s))")

    typer.echo(
        f"{stats['detections']} detection(s) over {stats['events']} event(s); "
        f"{stats['by_verification']}"
    )
    if stats["by_verification"].get("unverified"):
        typer.secho(
            "unverified detections are model claims, not facts — adjudicate before "
            "treating them as findings",
            fg=typer.colors.YELLOW,
        )


@eval_app.command("compare")
def eval_compare(
    run_a: int,
    run_b: int,
    include_machine: bool = typer.Option(False, help="Count machine labels."),
) -> None:
    """Paired comparison of two runs over the pairs they both judged."""
    from hontology.evalkit import compare

    with session_scope() as session:
        result = compare.compare_runs(session, run_a, run_b, include_machine=include_machine)

    typer.echo(f"shared labelled pairs  {result.n_shared_labelled}")
    typer.echo(
        f"discordant             {result.paired['discordant']} "
        f"(only A right: {result.paired['only_a_correct']}, "
        f"only B right: {result.paired['only_b_correct']})"
    )
    p = result.paired["p_value"]
    typer.echo(
        f"p-value                {p:.4f}" if p is not None else "p-value                —"
    )
    typer.secho(f"\n{result.verdict}", bold=True)


@eval_app.command("consistency")
def eval_consistency(run_id: int, against: int | None = None) -> None:
    """Determinism against another run, and agreement across documents."""
    from hontology.evalkit import compare

    with session_scope() as session:
        if against is not None:
            result = compare.determinism(session, run_id, against)
            status = "reproducible" if result["reproducible"] else "NOT reproducible"
            typer.secho(
                f"determinism: {status} "
                f"({result['n_shared']} shared, {result['flipped']} flipped)",
                fg=typer.colors.GREEN if result["reproducible"] else typer.colors.RED,
            )
            if not result["reproducible"]:
                typer.echo("  run-to-run noise is a floor under every A/B delta; a smaller")
                typer.echo("  difference than this cannot be attributed to a config change")
        agreement = compare.cross_document_agreement(session, run_id)
    typer.echo(
        f"cross-document: {agreement['multi_document_groups']} group(s), "
        f"{agreement['split_groups']} split, mean {agreement['mean_agreement']}"
    )


@eval_app.command("baseline")
def eval_baseline(
    run_id: int, out: Path = Path("baseline.json"), tolerance: float = 0.05
) -> None:
    """Record a run's metrics as the floor future runs must clear."""
    from hontology.evalkit import regression

    with session_scope() as session:
        baseline = regression.capture_baseline(session, run_id, tolerance=tolerance)
    regression.save_baseline(baseline, out)
    typer.echo(f"baseline from run {run_id} -> {out}")


@eval_app.command("gate")
def eval_gate(run_id: int, baseline_path: Path = Path("baseline.json")) -> None:
    """Regression gate: liveness first, then metric floors. Exits non-zero on failure."""
    from hontology.evalkit import regression

    baseline = regression.load_baseline(baseline_path)
    with session_scope() as session:
        result = regression.check(session, run_id, baseline)

    live = result["liveness"]
    typer.echo(f"liveness  {live['clean']}/{live['verdicts']} clean, {live['errors']} error(s)")
    for floor in result["floors"]:
        actual = floor["actual"]
        shown = f"{actual:.3f}" if actual is not None else "—"
        mark = "ok " if floor["ok"] else "FAIL"
        limit = f"{floor['limit']:.3f}" if floor.get("limit") is not None else "—"
        typer.echo(f"{mark}      {floor['metric']:<10} {shown}  (floor {limit})")

    if result["ok"]:
        typer.secho(
            f"\ngate passed ({result['n_judged_labelled']} labelled pairs checked)",
            fg=typer.colors.GREEN,
        )
    elif result["inconclusive"]:
        typer.secho("\ngate INCONCLUSIVE — nothing to check", fg=typer.colors.YELLOW)
        for failure in result["failures"]:
            typer.echo(f"  - {failure}")
        raise typer.Exit(2)
    else:
        typer.secho("\ngate FAILED", fg=typer.colors.RED)
        for failure in result["failures"]:
            typer.echo(f"  - {failure}")
        raise typer.Exit(1)


@eval_app.command("leaderboard")
def eval_leaderboard(limit: int = 50) -> None:
    """Recorded runs, ranked by F1, with intervals and denominators."""
    from hontology.evalkit import warehouse

    typer.echo(warehouse.format_leaderboard(warehouse.leaderboard(limit=limit)))


if __name__ == "__main__":
    app()
