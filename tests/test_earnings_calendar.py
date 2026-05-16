"""Tests for amms.data.earnings_calendar."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from amms.data.earnings_calendar import (
    EarningsEntry, add, check_positions, ensure_table, remove, upcoming,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    return conn


class TestEnsureTable:
    def test_creates_table(self):
        conn = sqlite3.connect(":memory:")
        ensure_table(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert any(t[0] == "earnings_calendar" for t in tables)

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        ensure_table(conn)
        ensure_table(conn)  # should not raise


class TestAdd:
    def test_add_basic(self):
        conn = _conn()
        assert add(conn, "AAPL", "2026-06-01") is True

    def test_add_with_time(self):
        conn = _conn()
        assert add(conn, "AAPL", "2026-06-01", time_of_day="after_close") is True

    def test_add_returns_false_bad_date(self):
        conn = _conn()
        assert add(conn, "AAPL", "not-a-date") is False

    def test_add_uppercase(self):
        conn = _conn()
        add(conn, "aapl", "2026-06-01")
        entries = upcoming(conn, within_days=999)
        assert any(e.symbol == "AAPL" for e in entries)

    def test_replace_existing(self):
        conn = _conn()
        add(conn, "AAPL", "2026-06-01", note="first")
        add(conn, "AAPL", "2026-06-01", note="updated")
        entries = upcoming(conn, within_days=999)
        aapl = [e for e in entries if e.symbol == "AAPL" and e.report_date == "2026-06-01"]
        assert len(aapl) == 1
        assert aapl[0].note == "updated"

    def test_invalid_time_defaults_to_unknown(self):
        conn = _conn()
        add(conn, "AAPL", "2026-06-01", time_of_day="morning")
        entries = upcoming(conn, within_days=999)
        aapl = next(e for e in entries if e.symbol == "AAPL")
        assert aapl.time_of_day == "unknown"


class TestRemove:
    def test_remove_specific_date(self):
        conn = _conn()
        add(conn, "AAPL", "2026-06-01")
        add(conn, "AAPL", "2026-09-01")
        n = remove(conn, "AAPL", "2026-06-01")
        assert n == 1
        entries = upcoming(conn, within_days=999)
        dates = [e.report_date for e in entries if e.symbol == "AAPL"]
        assert "2026-06-01" not in dates
        assert "2026-09-01" in dates

    def test_remove_all_for_symbol(self):
        conn = _conn()
        add(conn, "AAPL", "2026-06-01")
        add(conn, "AAPL", "2026-09-01")
        n = remove(conn, "AAPL")
        assert n == 2

    def test_remove_nonexistent_returns_zero(self):
        conn = _conn()
        assert remove(conn, "UNKN") == 0

    def test_remove_case_insensitive(self):
        conn = _conn()
        add(conn, "AAPL", "2026-06-01")
        n = remove(conn, "aapl")
        assert n == 1


class TestUpcoming:
    def test_returns_empty_if_none(self):
        conn = _conn()
        result = upcoming(conn, today=date(2026, 1, 1))
        assert result == []

    def test_returns_within_window(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "AAPL", "2026-01-15")  # 14 days away
        add(conn, "MSFT", "2026-03-01")  # outside 30-day window
        result = upcoming(conn, within_days=20, today=today)
        syms = [e.symbol for e in result]
        assert "AAPL" in syms
        assert "MSFT" not in syms

    def test_days_until_correct(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "AAPL", "2026-01-10")
        result = upcoming(conn, within_days=30, today=today)
        aapl = next(e for e in result if e.symbol == "AAPL")
        assert aapl.days_until == 9

    def test_past_dates_excluded(self):
        conn = _conn()
        today = date(2026, 1, 15)
        add(conn, "AAPL", "2026-01-01")  # past
        result = upcoming(conn, within_days=30, today=today)
        assert not any(e.symbol == "AAPL" for e in result)

    def test_filter_by_symbols(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "AAPL", "2026-01-10")
        add(conn, "MSFT", "2026-01-12")
        add(conn, "TSLA", "2026-01-14")
        result = upcoming(conn, within_days=30, symbols=["AAPL", "TSLA"], today=today)
        syms = {e.symbol for e in result}
        assert "AAPL" in syms
        assert "TSLA" in syms
        assert "MSFT" not in syms

    def test_sorted_by_date(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "B", "2026-01-20")
        add(conn, "A", "2026-01-10")
        result = upcoming(conn, within_days=30, today=today)
        dates = [e.report_date for e in result]
        assert dates == sorted(dates)


class TestCheckPositions:
    def test_flags_upcoming_earnings(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "AAPL", "2026-01-05")
        result = check_positions(conn, ["AAPL", "MSFT"], within_days=7, today=today)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_no_flags_if_outside_window(self):
        conn = _conn()
        today = date(2026, 1, 1)
        add(conn, "AAPL", "2026-02-15")
        result = check_positions(conn, ["AAPL"], within_days=7, today=today)
        assert result == []

    def test_returns_empty_bad_table(self):
        conn = sqlite3.connect(":memory:")
        result = check_positions(conn, ["AAPL"])
        assert result == []
