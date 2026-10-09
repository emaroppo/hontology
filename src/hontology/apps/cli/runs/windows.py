"""Walking a calendar's windows: which are ready, and processing each into a run."""

from __future__ import annotations

from dataclasses import dataclass

import typer

from hontology.apps.cli.common import load_run
from hontology.db.session import session_scope
from hontology.pipeline.ingest.feed import slices


@dataclass
class CalendarWalk:
    """One pass over a calendar's windows into one run, and how far it has got."""

    run_id: int
    entries: list
    loci: dict
    before: int
    after: int
    budget: int | None
    prepare_only: bool
    candidates_from: int | None
    # Entries finished by this kind of pass, and windows set aside as unscorable.
    finished: set[str]
    unobservable: set[str]

    @property
    def progress_key(self) -> str:
        # A prepare-only pass tracks its own progress, so a later judging pass
        # still visits every window.
        return "calendar_prepared" if self.prepare_only else "calendar_done"

    def pending(self) -> bool:
        return len(self.finished | self.unobservable) < len(self.entries)

    def visit(self, entry) -> bool:
        """Process `entry`'s window if it is ready; whether anything happened."""
        places = [self.loci[c] for c in entry.countries]
        ready = window_ready(entry, places, self.before, self.after)
        if ready and self._set_aside_if_unobservable(entry, places):
            return True
        if ready and self.candidates_from is not None:
            # Reusing another run's retrieval: wait until it has
            # finished this window, so the documents are all there.
            with session_scope() as session:
                source = load_run(session, self.candidates_from)
                ready = entry.id in (source.manifest or {}).get("calendar_done", [])
        if not ready:
            return False
        self._process(entry, places)
        return True

    def _set_aside_if_unobservable(self, entry, places) -> bool:
        """Windows the source published nothing for are recorded and never scored."""
        from hontology.evaluation.calendar import runner as calendar_run

        with session_scope() as session:
            if not calendar_run.window_unobservable(
                session, entry, places, before=self.before, after=self.after
            ):
                return False
            self.unobservable.add(entry.id)
            run = load_run(session, self.run_id)
            stored = set((run.manifest or {}).get("calendar_unobservable", []))
            run.manifest = (run.manifest or {}) | {
                "calendar_unobservable": sorted(stored | {entry.id})
            }
            typer.echo(
                f"[set aside] {entry.id}: the source published nothing "
                "for this window; not scored"
            )
        return True

    def _process(self, entry, places) -> None:
        from hontology.evaluation.calendar import runner as calendar_run

        with session_scope() as session:
            run = load_run(session, self.run_id)
            summary = calendar_run.process_entry(
                session,
                run,
                entry,
                places,
                before=self.before,
                after=self.after,
                budget=self.budget,
                judge=not self.prepare_only,
                candidates_from=self.candidates_from,
            )
            self.finished.add(entry.id)
            run.manifest = (run.manifest or {}) | {self.progress_key: sorted(self.finished)}
        judge = summary["judge"] or {}
        judge_note = (
            f"{summary['pairs_selected']} pair(s) selected"
            if self.prepare_only
            else f"judged {judge.get('judged', 0)}, matched {judge.get('matched', 0)}"
        )
        typer.echo(
            f"[{len(self.finished)}/{len(self.entries)}] {entry.id}: "
            f"{summary['passed_filter']} in scope, "
            f"{summary['representatives']} unique, {judge_note}"
        )


def window_ready(entry, places, before: int, after: int) -> bool:
    """Whether a window's feed is all in, re-ingesting its failed slices once."""
    from hontology.evaluation.calendar import runner as calendar_run

    with session_scope() as session:
        ready, failed = calendar_run.window_status(
            session, entry, places, before=before, after=after
        )
    if ready or not failed:
        return ready
    for feed, key in failed:
        with session_scope() as session:
            scope = places if feed == slices.FEED_GKG else None
            slices.ingest_slice(session, key, feed=feed, loci=scope)
    with session_scope() as session:
        ready, _ = calendar_run.window_status(
            session, entry, places, before=before, after=after
        )
    return ready
