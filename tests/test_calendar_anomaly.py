"""Tests for amms.analysis.calendar_anomaly."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.calendar_anomaly import CalendarAnomalyReport, CalendarSlice, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, entered_at TEXT, status TEXT)")
    conn.executemany("INSERT INTO trades VALUES (?, ?, 'closed')", rows)
    conn.commit()
    return conn


def _spread_rows(n: int = 60) -> list[tuple]:
    """Trades spread across different dates and days."""
    rows = []
    day = 1
    month = 1
    for i in range(n):
        # Cycle through days 1-28, months 1-12
        ts = f"2024-{month:02d}-{day:02d} 10:00:00"
        pnl = 1.5 if i % 3 != 0 else -1.0
        rows.append((pnl, ts))
        day = (day % 27) + 1
        if day == 1:
            month = (month % 12) + 1
    return rows


def _monday_biased(n: int = 60) -> list[tuple]:
    """Mondays are always winners, Fridays always losers."""
    rows = []
    # 2024-01-01 is Monday
    for i in range(n):
        day_offset = i % 5
        # week 1 = Jan 1-5, week 2 = Jan 8-12, etc.
        week = i // 5
        day = 1 + week * 7 + day_offset
        # Keep in a single month
        if day > 28:
            break
        ts = f"2024-01-{day:02d} 10:00:00"
        pnl = 2.0 if day_offset == 0 else -1.5  # Monday=2, others=-1.5
        rows.append((pnl, ts))
    return rows


class TestEdgeCases:
    def test_returns_none_empty_db(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(1.0, "2024-01-01 10:00") for _ in range(5)]
        conn = _make_db(rows)
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_db(_spread_rows(40))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, CalendarAnomalyReport)


class TestWeekdaySlices:
    def test_five_weekday_slices(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        assert len(result.by_weekday) == 5

    def test_weekday_labels(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        labels = [s.label for s in result.by_weekday]
        assert "Monday" in labels
        assert "Friday" in labels

    def test_win_rate_in_range(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        for s in result.by_weekday:
            if s.n_trades > 0:
                assert 0.0 <= s.win_rate <= 100.0

    def test_slice_is_calendar_slice(self):
        conn = _make_db(_spread_rows(40))
        result = compute(conn)
        assert result is not None
        for s in result.by_weekday:
            assert isinstance(s, CalendarSlice)


class TestMonthSlices:
    def test_twelve_month_slices(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        assert len(result.by_month) == 12

    def test_month_labels_present(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        labels = [s.label for s in result.by_month]
        assert "Jan" in labels
        assert "Dec" in labels


class TestQEndEffect:
    def test_qend_and_other_returned(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        qend, other = result.qend_vs_other
        assert isinstance(qend, CalendarSlice)
        assert isinstance(other, CalendarSlice)

    def test_qend_label(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        qend, _ = result.qend_vs_other
        assert "Q-end" in qend.label


class TestMonthStartEffect:
    def test_month_start_and_other_returned(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        ms, other = result.month_start_vs_other
        assert isinstance(ms, CalendarSlice)
        assert isinstance(other, CalendarSlice)


class TestBestWorst:
    def test_best_weekday_is_calendarslice(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        if result.best_weekday:
            assert isinstance(result.best_weekday, CalendarSlice)

    def test_best_ge_worst_weekday(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        if result.best_weekday and result.worst_weekday:
            assert result.best_weekday.avg_pnl_pct >= result.worst_weekday.avg_pnl_pct - 0.01

    def test_best_month_set_when_trades_exist(self):
        conn = _make_db(_spread_rows(60))
        result = compute(conn)
        assert result is not None
        # Spread across months → at least some months have trades
        assert result.best_month is not None or result.n_trades < 12


class TestMetadata:
    def test_n_trades_correct(self):
        rows = _spread_rows(40)
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 40


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_spread_rows(40))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_calendar(self):
        conn = _make_db(_spread_rows(40))
        result = compute(conn)
        assert result is not None
        assert "calendar" in result.verdict.lower() or "anomal" in result.verdict.lower() or "pattern" in result.verdict.lower()
