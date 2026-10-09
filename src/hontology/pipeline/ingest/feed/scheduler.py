"""Optional continuous ingest.

**Continuous monitoring is opt-in.** Nothing here runs unless someone explicitly
starts it — the API never launches a scheduler, and the default `docker compose
up` does not bring a watcher with it. One-shot ingest and explicit backfill are
the baseline; this module only adds "keep doing that on a schedule".

That split matters for a system meant to be picked up and put down: a clone
should not start crawling the internet on its own, and a developer poking at the
ontology should not need a background process running to do it.

When it *is* running, three things keep it honest:

- **Aligned polling.** Waking on a fixed interval drifts out of phase with a feed
  that publishes on the quarter hour, so wake-ups land at the least useful moment.
  The loop targets each boundary plus a grace period instead.
- **An advisory lock.** Two watchers, or a watcher racing a manual catch-up,
  would both download the same slice. The lock makes the second one a no-op
  rather than a duplicate.
- **Graceful shutdown.** SIGTERM finishes the slice in flight and commits it,
  instead of losing the work and leaving a half-written status row.
"""

from __future__ import annotations

import logging
import signal
import threading
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from hontology.db.session import session_scope
from hontology.pipeline.ingest.feed import catchup, gdelt

log = logging.getLogger(__name__)

# Arbitrary but fixed: the key for the Postgres advisory lock that serializes
# ingest. Any process taking this lock is the one allowed to fetch slices.
INGEST_LOCK_KEY = 0x484F4E54  # "HONT"


def try_lock(session: Session, key: int = INGEST_LOCK_KEY) -> bool:
    """Take the session-scoped ingest lock, or report that someone else holds it.

    Session-scoped rather than transaction-scoped so it is held for the whole
    slice, and released automatically if the process dies.
    """
    return bool(session.scalar(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}))


def unlock(session: Session, key: int = INGEST_LOCK_KEY) -> None:
    session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})


class Watcher:
    """Runs `catch_up` on the feed's publish cadence until asked to stop."""

    def __init__(
        self,
        *,
        grace_seconds: float = 90.0,
        max_slices: int = 32,
        max_backoff: float = 900.0,
    ) -> None:
        self.grace_seconds = grace_seconds
        self.max_slices = max_slices
        self.max_backoff = max_backoff
        self._stop = threading.Event()
        self._failures = 0

    def request_stop(self, *_: object) -> None:
        log.info("stop requested; finishing the current slice")
        self._stop.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep, so shutdown does not wait out a 15-minute nap."""
        self._stop.wait(seconds)

    def _backoff(self) -> float:
        """Exponential backoff, capped, after a failed pass.

        The feed being briefly unavailable is normal. Retrying it every few
        seconds turns a transient outage into a self-inflicted hammering.
        """
        return min(self.max_backoff, 30.0 * (2 ** min(self._failures, 5)))

    def run_once(self) -> dict | None:
        """One pass. Returns None when another process holds the lock."""
        with session_scope() as session:
            if not try_lock(session):
                log.debug("another process holds the ingest lock; skipping this pass")
                return None
            try:
                return catchup.catch_up(session, max_slices=self.max_slices)
            finally:
                unlock(session)

    def run(self) -> None:
        log.info("ingest watcher started (opt-in); polling on the feed's 15-minute cadence")
        while not self._stop.is_set():
            try:
                result = self.run_once()
                self._failures = 0
                if result is not None:
                    log.info(
                        "pass complete: watermark=%s processed=%s remaining=%s",
                        result["watermark"],
                        result["processed"],
                        result["remaining"],
                    )
                    # Still behind: go again immediately rather than idling a
                    # quarter hour while a backlog sits there.
                    if result["remaining"] > 0 and not self._stop.is_set():
                        continue
            except Exception as exc:  # noqa: BLE001 - a watcher must not die on one bad pass
                self._failures += 1
                delay = self._backoff()
                log.warning("ingest pass failed (%s); backing off %.0fs", exc, delay)
                self._sleep(delay)
                continue

            self._sleep(
                gdelt.seconds_until_next_slice(
                    datetime.now(UTC), grace_seconds=self.grace_seconds
                )
            )

        log.info("ingest watcher stopped")
