"""Tests for amms.analysis.journal_summary."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.journal_summary import JournalSummary, PeriodStats, compute


def _make_conn(rows: list[tuple[str, float]]) -> sqlite3.Connection:
    """rows: (sell_ts, pnl)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (ts, pnl) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', "
            "'2026-01-01T10:00:00', ?, 100.0, 101.0, 10.0, ?)",
            (i, ts, pnl),
        )
    conn.commit()
    return conn


def _jan_trades(pnls: list[float]) -> list[tuple[str, float]]:
    return [(f"2026-01-{i+1:02d}T15:00:00", p) for i, p in enumerate(pnls)]


def _feb_trades(pnls: list[float]) -> list[tuple[str, float]]:
    return [(f"2026-02-{i+1:02d}T15:00:00", p) for i, p in enumerate(pnls)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn(_jan_trades([50.0, 50.0]))
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_jan_trades([50.0, -30.0, 50.0, -30.0, 80.0]))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, JournalSummary)


class TestMonthlyMode:
    def test_two_months_two_periods(self):
        rows = _jan_trades([50.0, -30.0, 50.0]) + _feb_trades([80.0, -20.0, 60.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        assert result.n_periods == 2

    def test_period_keys_monthly(self):
        rows = _jan_trades([50.0, -30.0, 50.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        periods = [p.period for p in result.periods]
        assert "2026-01" in periods

    def test_pnl_correct(self):
        rows = _jan_trades([50.0, -30.0, 80.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        jan = result.periods[0]
        assert jan.total_pnl == pytest.approx(100.0, abs=0.1)

    def test_win_rate_correct(self):
        rows = _jan_trades([50.0, 50.0, -30.0, 50.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        jan = result.periods[0]
        assert jan.win_rate == pytest.approx(75.0, abs=0.1)

    def test_profit_factor_correct(self):
        rows = _jan_trades([100.0, -50.0, 100.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        jan = result.periods[0]
        assert jan.profit_factor is not None
        assert jan.profit_factor == pytest.approx(4.0, abs=0.1)

    def test_profit_factor_none_no_losses(self):
        rows = _jan_trades([50.0, 60.0, 70.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        jan = result.periods[0]
        assert jan.profit_factor is None

    def test_best_worst_trade(self):
        rows = _jan_trades([200.0, -30.0, 50.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        jan = result.periods[0]
        assert jan.best_trade == pytest.approx(200.0, abs=0.1)
        assert jan.worst_trade == pytest.approx(-30.0, abs=0.1)


class TestWeeklyMode:
    def test_weekly_mode(self):
        # Multiple weeks in January 2026
        rows = [
            ("2026-01-05T15:00:00", 50.0),  # week 1
            ("2026-01-06T15:00:00", -30.0),
            ("2026-01-12T15:00:00", 80.0),  # week 2
            ("2026-01-13T15:00:00", 40.0),
            ("2026-01-19T15:00:00", -20.0),  # week 3
        ]
        conn = _make_conn(rows)
        result = compute(conn, mode="weekly")
        assert result is not None
        assert result.mode == "weekly"
        assert result.n_periods >= 2

    def test_weekly_key_format(self):
        rows = [("2026-01-05T15:00:00", 50.0)] * 3
        conn = _make_conn(rows)
        result = compute(conn, mode="weekly")
        assert result is not None
        period = result.periods[0].period
        assert "W" in period


class TestOverallStats:
    def test_best_period(self):
        rows = _jan_trades([200.0, 200.0, -10.0]) + _feb_trades([-50.0, -50.0, -10.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        assert result.best_period == "2026-01"

    def test_worst_period(self):
        rows = _jan_trades([200.0, 200.0, -10.0]) + _feb_trades([-50.0, -50.0, -10.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        assert result.worst_period == "2026-02"

    def test_overall_pnl(self):
        rows = _jan_trades([50.0, -30.0, 80.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        assert result.overall_pnl == pytest.approx(100.0, abs=0.1)

    def test_n_trades_correct(self):
        rows = _jan_trades([50.0, -30.0, 80.0]) + _feb_trades([40.0, -20.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        assert result.n_trades == 5

    def test_limit_respected(self):
        rows = _jan_trades([50.0] * 20)
        conn = _make_conn(rows)
        result = compute(conn, limit=10)
        assert result is not None
        assert result.n_trades == 10

    def test_periods_chronological(self):
        rows = _jan_trades([50.0, -30.0, 50.0]) + _feb_trades([80.0, -20.0, 60.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="monthly")
        assert result is not None
        periods = [p.period for p in result.periods]
        assert periods == sorted(periods)

    def test_invalid_mode_defaults_monthly(self):
        rows = _jan_trades([50.0, -30.0, 80.0])
        conn = _make_conn(rows)
        result = compute(conn, mode="invalid")
        assert result is not None
        assert result.mode == "monthly"
