"""Slice arithmetic — the part that has to be exactly right for auto-update.

All pure functions, no network. These cover the cases that make a continuously
running ingester drift, double-ingest, or silently stop catching up.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from hontology.pipeline.ingest.feed import gdelt


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


class TestSliceKey:
    @pytest.mark.parametrize(
        ("moment", "expected"),
        [
            (utc(2026, 8, 26, 12, 0, 0), "20260826120000"),
            (utc(2026, 8, 26, 12, 14, 59), "20260826120000"),
            (utc(2026, 8, 26, 12, 15, 0), "20260826121500"),
            (utc(2026, 8, 26, 12, 44, 59), "20260826123000"),
            (utc(2026, 8, 26, 12, 45, 1), "20260826124500"),
            (utc(2026, 8, 26, 23, 59, 59), "20260826234500"),
        ],
    )
    def test_floors_to_the_quarter_hour(self, moment, expected):
        assert gdelt.slice_key(moment) == expected

    def test_non_utc_input_is_converted_not_truncated(self):
        """A local timestamp must convert to UTC, not have its clock face read.

        14:20 in a +02:00 zone is 12:20 UTC, so it belongs to the 12:15 slice —
        not the 14:15 one its digits suggest.
        """
        rome = datetime(2026, 8, 26, 14, 20, tzinfo=timezone(timedelta(hours=2)))
        assert gdelt.slice_key(rome) == "20260826121500"

    def test_round_trips_through_datetime(self):
        key = "20260826121500"
        assert gdelt.slice_key(gdelt.key_to_datetime(key)) == key


class TestNeighbours:
    def test_next_and_previous(self):
        assert gdelt.next_key("20260826121500") == "20260826123000"
        assert gdelt.previous_key("20260826121500") == "20260826120000"

    def test_next_crosses_the_hour(self):
        assert gdelt.next_key("20260826124500") == "20260826130000"

    def test_next_crosses_midnight(self):
        assert gdelt.next_key("20260826234500") == "20260827000000"

    def test_previous_crosses_midnight(self):
        assert gdelt.previous_key("20260827000000") == "20260826234500"

    def test_next_crosses_a_month_boundary(self):
        assert gdelt.next_key("20260831234500") == "20260901000000"


class TestKeysBetween:
    def test_excludes_the_watermark_and_includes_the_target(self):
        """The watermark is already done; re-including it would double-ingest."""
        keys = gdelt.keys_between("20260826120000", "20260826124500")
        assert keys == ["20260826121500", "20260826123000", "20260826124500"]

    def test_caught_up_returns_nothing(self):
        assert gdelt.keys_between("20260826121500", "20260826121500") == []

    def test_a_watermark_ahead_of_the_target_returns_nothing(self):
        """Clock skew or a republished slice must not produce a negative range."""
        assert gdelt.keys_between("20260826123000", "20260826120000") == []

    def test_overnight_outage_is_fully_enumerated(self):
        keys = gdelt.keys_between("20260826120000", "20260827120000")
        assert len(keys) == 96  # 24h at 4 slices/hour
        assert keys[0] == "20260826121500"
        assert keys[-1] == "20260827120000"

    def test_limit_bounds_a_long_catch_up(self):
        """A month of downtime must not turn into one unbounded crawl."""
        keys = gdelt.keys_between("20260701000000", "20260826120000", limit=10)
        assert len(keys) == 10
        assert keys[0] == "20260701001500"


class TestPollAlignment:
    def test_sleeps_until_just_after_the_next_boundary(self):
        # 12:01:00 → next boundary 12:15:00 is 14m00s away, plus 90s grace.
        delay = gdelt.seconds_until_next_slice(utc(2026, 8, 26, 12, 1, 0), grace_seconds=90)
        assert delay == pytest.approx(14 * 60 + 90)

    def test_just_after_a_boundary_waits_for_the_following_one(self):
        # 12:15:05 → next boundary 12:30:00 is 14m55s away, plus 90s grace.
        delay = gdelt.seconds_until_next_slice(utc(2026, 8, 26, 12, 15, 5), grace_seconds=90)
        assert delay == pytest.approx(14 * 60 + 55 + 90)

    def test_inside_the_grace_window_still_returns_a_positive_delay(self):
        """Never return 0 or negative — that would spin the loop."""
        for second in range(0, 60):
            delay = gdelt.seconds_until_next_slice(
                utc(2026, 8, 26, 12, 14, second), grace_seconds=90
            )
            assert delay > 0

    def test_delay_never_exceeds_one_interval_plus_grace(self):
        for minute in range(60):
            delay = gdelt.seconds_until_next_slice(
                utc(2026, 8, 26, 12, minute, 0), grace_seconds=90
            )
            assert 0 < delay <= gdelt.SLICE_INTERVAL.total_seconds() + 90


class TestLag:
    def test_never_ingested_is_unknown_not_zero(self):
        """Reporting 0 for a fresh install would look healthy when it is idle."""
        assert gdelt.lag_slices(None, utc(2026, 8, 26, 12, 0)) is None

    def test_up_to_date_is_zero(self):
        assert gdelt.lag_slices("20260826120000", utc(2026, 8, 26, 12, 5)) == 0

    def test_counts_missed_slices(self):
        assert gdelt.lag_slices("20260826120000", utc(2026, 8, 26, 13, 0)) == 4


class TestExportUrl:
    def test_builds_the_zip_url(self):
        assert gdelt.export_url("http://example.test/gdeltv2", "20260826121500") == (
            "http://example.test/gdeltv2/20260826121500.export.CSV.zip"
        )

    def test_tolerates_a_trailing_slash(self):
        assert gdelt.export_url("http://example.test/gdeltv2/", "20260826121500").endswith(
            "/20260826121500.export.CSV.zip"
        )
