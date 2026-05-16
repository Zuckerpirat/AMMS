"""Tests for amms.analysis.profit_calendar."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from amms.analysis.profit_calendar import DayStats, MonthStats, ProfitCalendarReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (sell_ts, pnl)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (sell_ts, pnl) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', '2026-01-01T09:00:00', ?, 100.0, 105.0, 10.0, ?)",
            (i, sell_ts, pnl),
        )
    conn.commit()
    return conn


def _rows(n: int = 20) -> list[tuple]:
    """20 rows across January 2026."""
    result = []
    for i in range(n):
        day = (i % 20) + 1
        pnl = 50.0 if i % 2 == 0 else -30.0
        result.append((f"2026-01-{day:02d}T14:00:00", pnl))
    return result


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [("2026-01-05T14:00:00", 50.0), ("2026-01-06T14:00:00", -30.0)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_rows(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, ProfitCalendarReport)


class TestMonths:
    def test_has_months(self):
        conn = _make_conn(_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.n_months >= 1

    def test_month_has_days(self):
        conn = _make_conn(_rows(20))
        result = compute(conn)
        assert result is not None
        for m in result.months:
            assert m.n_trading_days > 0
            assert len(m.days) == m.n_trading_days

    def test_two_months_detected(self):
        rows = []
        for i in range(10):
            rows.append((f"2026-01-{i+1:02d}T14:00:00", 50.0 if i % 2 == 0 else -30.0))
        for i in range(10):
            rows.append((f"2026-02-{i+1:02d}T14:00:00", 50.0 if i % 2 == 0 else -30.0))
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_months == 2

    def test_profitable_days_count(self):
        rows = [
            ("2026-01-05T14:00:00", 100.0),
            ("2026-01-06T14:00:00", -50.0),
            ("2026-01-07T14:00:00", 80.0),
            ("2026-01-08T14:00:00", -20.0),
            ("2026-01-09T14:00:00", 60.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall_profitable_days == 3
        assert result.overall_losing_days == 2

    def test_month_pnl_sum_correct(self):
        pnls = [100.0, -50.0, 80.0, -20.0, 60.0, -10.0, 40.0]
        rows = [(f"2026-01-{i+1:02d}T14:00:00", p) for i, p in enumerate(pnls)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall_pnl == pytest.approx(sum(pnls), abs=0.01)


class TestBestWorst:
    def test_best_day_identified(self):
        rows = [
            ("2026-01-05T14:00:00", 200.0),
            ("2026-01-06T14:00:00", 50.0),
            ("2026-01-07T14:00:00", -30.0),
            ("2026-01-08T14:00:00", -10.0),
            ("2026-01-09T14:00:00", 80.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.best_day == date(2026, 1, 5)

    def test_worst_day_identified(self):
        rows = [
            ("2026-01-05T14:00:00", 200.0),
            ("2026-01-06T14:00:00", 50.0),
            ("2026-01-07T14:00:00", -300.0),
            ("2026-01-08T14:00:00", -10.0),
            ("2026-01-09T14:00:00", 80.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.worst_day == date(2026, 1, 7)

    def test_multiple_trades_same_day(self):
        rows = [
            ("2026-01-05T10:00:00", 100.0),
            ("2026-01-05T14:00:00", 50.0),
            ("2026-01-06T14:00:00", -200.0),
            ("2026-01-07T14:00:00", 30.0),
            ("2026-01-08T14:00:00", 20.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        jan5_month = result.months[0]
        jan5_day = next(d for d in jan5_month.days if d.date == date(2026, 1, 5))
        assert jan5_day.n_trades == 2
        assert jan5_day.total_pnl == pytest.approx(150.0, abs=0.01)


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_limit_respected(self):
        rows = [(f"2026-01-{(i%20)+1:02d}T14:00:00", 50.0) for i in range(50)]
        conn = _make_conn(rows)
        result = compute(conn, limit=10)
        assert result is not None
        total = result.overall_profitable_days + result.overall_losing_days
        assert total <= 10
