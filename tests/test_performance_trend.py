"""Tests for amms.analysis.performance_trend."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.performance_trend import MonthlyPnL, PerformanceTrendReport, compute


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
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', ?, 100.0, 105.0, 10.0, ?)",
            (i, sell_ts, pnl),
        )
    conn.commit()
    return conn


def _make_improving_rows() -> list[tuple]:
    """PnL improves each month: Jan=100, Feb=200, Mar=300, Apr=400."""
    rows = []
    for month, base in [(1, 100.0), (2, 200.0), (3, 300.0), (4, 400.0)]:
        for day in range(1, 6):
            rows.append((f"2026-{month:02d}-{day:02d}T14:00:00", base / 5))
    return rows


def _make_declining_rows() -> list[tuple]:
    """PnL declines each month: Jan=400, Feb=300, Mar=200, Apr=100."""
    rows = []
    for month, base in [(1, 400.0), (2, 300.0), (3, 200.0), (4, 100.0)]:
        for day in range(1, 6):
            rows.append((f"2026-{month:02d}-{day:02d}T14:00:00", base / 5))
    return rows


def _make_mixed_rows(n_months: int = 4) -> list[tuple]:
    rows = []
    for month in range(1, n_months + 1):
        pnl = 100.0 if month % 2 == 0 else -50.0
        for day in range(1, 4):
            rows.append((f"2026-{month:02d}-{day:02d}T14:00:00", pnl))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few_months(self):
        rows = [(f"2026-01-0{i}T14:00:00", 50.0) for i in range(1, 5)]
        conn = _make_conn(rows)
        assert compute(conn) is None  # only 1 month

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, PerformanceTrendReport)


class TestTrend:
    def test_improving_detected(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert result.trend_direction == "improving"
        assert result.slope > 0

    def test_declining_detected(self):
        conn = _make_conn(_make_declining_rows())
        result = compute(conn)
        assert result is not None
        assert result.trend_direction == "declining"
        assert result.slope < 0

    def test_direction_valid(self):
        conn = _make_conn(_make_mixed_rows())
        result = compute(conn)
        assert result is not None
        assert result.trend_direction in ("improving", "declining", "flat")

    def test_r_squared_bounded(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.r_squared <= 1.0

    def test_r_squared_high_for_linear(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert result.r_squared > 0.9


class TestMonthly:
    def test_monthly_count(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert result.n_months == 4
        assert len(result.monthly) == 4

    def test_monthly_labels(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        for m in result.monthly:
            assert len(m.label) == 7  # "YYYY-MM"
            assert "-" in m.label

    def test_best_worst_month(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert result.best_month == "2026-04"
        assert result.worst_month == "2026-01"

    def test_consistency_range(self):
        conn = _make_conn(_make_mixed_rows())
        result = compute(conn)
        assert result is not None
        assert 0 <= result.consistency_pct <= 100


class TestAcceleration:
    def test_acceleration_valid(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert result.acceleration in ("accelerating", "decelerating", "stable")

    def test_accelerating_detected(self):
        """Second half much better than first."""
        rows = []
        # Jan/Feb: small profit
        for month in [1, 2]:
            for d in range(1, 4):
                rows.append((f"2026-{month:02d}-{d:02d}T14:00:00", 10.0))
        # Mar/Apr: large profit
        for month in [3, 4]:
            for d in range(1, 4):
                rows.append((f"2026-{month:02d}-{d:02d}T14:00:00", 500.0))
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.acceleration == "accelerating"


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_make_improving_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
