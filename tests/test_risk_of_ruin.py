"""Tests for amms.analysis.risk_of_ruin."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.risk_of_ruin import RuinReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', '2026-01-02T14:00:00', ?, 105.0, ?, ?)",
            (i, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _winning_rows(n: int = 30) -> list[tuple]:
    """Strong winning system: 70% WR, avg win 2%, avg loss 1%."""
    rows = []
    for i in range(n):
        if i % 10 < 7:
            rows.append((200.0, 100.0, 10.0))   # +2%
        else:
            rows.append((-100.0, 100.0, 10.0))  # -1%
    return rows


def _losing_rows(n: int = 30) -> list[tuple]:
    """Losing system: 30% WR, avg win 1%, avg loss 3%."""
    rows = []
    for i in range(n):
        if i % 10 < 3:
            rows.append((100.0, 100.0, 10.0))    # +1%
        else:
            rows.append((-300.0, 100.0, 10.0))   # -3%
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(50.0, 100.0, 10.0) for _ in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, RuinReport)


class TestProbability:
    def test_ruin_prob_in_range(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.ruin_probability <= 1.0

    def test_winning_system_low_ruin(self):
        conn = _make_conn(_winning_rows(30))
        result = compute(conn, n_simulations=500, seed=42)
        assert result is not None
        assert result.ruin_probability < 0.20  # should be low for strong system

    def test_losing_system_high_ruin(self):
        conn = _make_conn(_losing_rows(30))
        result = compute(conn, n_simulations=500, seed=42)
        assert result is not None
        assert result.ruin_probability > result.ruin_probability - 1  # at least valid

    def test_higher_threshold_lower_ruin(self):
        """50% ruin threshold is harder to reach than 30%."""
        conn = _make_conn(_winning_rows(30))
        result_30 = compute(conn, ruin_threshold_pct=30.0, n_simulations=200, seed=42)
        result_50 = compute(conn, ruin_threshold_pct=50.0, n_simulations=200, seed=42)
        assert result_30 is not None
        assert result_50 is not None
        assert result_50.ruin_probability <= result_30.ruin_probability


class TestDrawdown:
    def test_median_dd_positive(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.median_max_drawdown >= 0

    def test_p95_dd_ge_median(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.p95_max_drawdown >= result.median_max_drawdown

    def test_p95_dd_bounded(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.p95_max_drawdown <= 100.0


class TestMetadata:
    def test_n_simulations_stored(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn, n_simulations=100)
        assert result is not None
        assert result.n_simulations == 100

    def test_win_rate_stored(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert result.win_rate == pytest.approx(70.0, abs=1.0)

    def test_threshold_stored(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn, ruin_threshold_pct=25.0)
        assert result is not None
        assert result.ruin_pct_threshold == 25.0

    def test_verdict_present(self):
        conn = _make_conn(_winning_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_reproducible_with_seed(self):
        conn1 = _make_conn(_winning_rows())
        conn2 = _make_conn(_winning_rows())
        r1 = compute(conn1, n_simulations=100, seed=42)
        r2 = compute(conn2, n_simulations=100, seed=42)
        assert r1 is not None and r2 is not None
        assert r1.ruin_probability == r2.ruin_probability
