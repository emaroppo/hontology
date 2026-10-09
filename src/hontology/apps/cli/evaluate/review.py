"""`hontology eval calendar-review-*`: a person's review of calendar matches."""

from __future__ import annotations

from pathlib import Path

import typer

from hontology.apps.cli.common import After, Before, write_csv
from hontology.db.session import session_scope

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


def eval_calendar_review_export(
    run_id: int,
    calendar_path: Path,
    out: Path = typer.Option(..., help="CSV to write; fill `confirmed` with yes or no."),
    per_entry: int = typer.Option(5, help="Unreviewed matches listed per entry."),
    before: Before = 1,
    after: After = 2,
) -> None:
    """Write the matches still awaiting review, earliest first, for a person to mark.

    For an event or precursor, `confirmed` asks: does this article describe this
    very event? For a control: does it report a real instance of the concept
    there and then? Entries already confirmed are left out.
    """
    from hontology.evaluation.calendar import events as calendar
    from hontology.evaluation.calendar import score as calendar_score

    entries = calendar.load(calendar_path)
    by_id = {e.id: e for e in entries}
    with session_scope() as session:
        result = calendar_score.evaluate(session, run_id, entries, before=before, after=after)
    rows = [
        {
            **{key: row[key] for key in ("kind", "concept", "countries", "date")},
            "entry_id": row["id"],
            "description": by_id[row["id"]].description,
            "first_seen": match["first_seen"],
            "url": match["url"],
            "evidence": (match["evidence"] or "").replace("\n", " "),
            "confirmed": "",
            "note": "",
        }
        for row in result["entries"]
        if row["verified"] is None
        for match in [m for m in row["matches"] if m["confirmed"] is None][:per_entry]
    ]
    write_csv(out, REVIEW_COLUMNS, rows)
    typer.echo(
        f"{len(rows)} match(es) to review across {len({r['entry_id'] for r in rows})} entries"
    )


def eval_calendar_review_import(path: Path) -> None:
    """Read reviewed matches back; rows left blank in `confirmed` are skipped."""
    import csv

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from hontology.db.models import CalendarReview

    answers = dict.fromkeys(("yes", "y", "true", "1"), True)
    answers |= dict.fromkeys(("no", "n", "false", "0"), False)
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
