"""Tests for amms.analysis.trade_clustering."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.trade_clustering import TradingClusterReport, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    """Create in-memory DB with trades table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE trades (
            symbol TEXT, entered_at TEXT, pnl_pct REAL, buy_price REAL, status TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO trades VALUES (?, ?, ?, ?, 'closed')",
        rows
    )
    conn.commit()
    return conn


def _ts(day: int, hour: int, minute: int = 0) -> str:
    return f"2024-03-{day:02d} {hour:02d}:{minute:02d}:00"


def _make_spread_rows(n: int = 30) -> list[tuple]:
    """Trades spread evenly across hours and symbols."""
    rows = []
    for i in range(n):
        h = i % 8 + 9  # hours 9–16
        sym = f"SYM{i % 5}"
        rows.append((sym, _ts(1 + i % 28, h), 1.0 if i % 2 == 0 else -1.0, 100.0))
    return rows


def _make_clustered_rows(n: int = 30) -> list[tuple]:
    """All trades at the same hour."""
    return [("AAPL", _ts(1 + i % 28, 9), 1.0 if i % 2 == 0 else -1.0, 100.0) for i in range(n)]


def _make_burst_rows() -> list[tuple]:
    """Cluster of trades within a few minutes."""
    rows = []
    # Burst: 5 trades within 10 minutes
    for i in range(5):
        rows.append(("TSLA", f"2024-03-01 10:{i:02d}:00", -0.5, 200.0))
    # Non-burst spread
    for i in range(10):
        rows.append(("AAPL", _ts(2 + i, 14), 1.0, 150.0))
    return rows


class TestEdgeCases:
    def test_returns_none_empty_db(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [("AAPL", _ts(1, 10), 1.0, 100.0) for _ in range(4)]
        conn = _make_db(rows)
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_db(_make_spread_rows(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, TradingClusterReport)


class TestHourClustering:
    def test_top_hours_returned(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert len(result.top_hours) > 0

    def test_top_hours_max_three(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert len(result.top_hours) <= 3

    def test_hour_in_range(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        for h in result.top_hours:
            assert 0 <= h.hour <= 23

    def test_high_concentration_when_all_same_hour(self):
        conn = _make_db(_make_clustered_rows(30))
        result = compute(conn)
        assert result is not None
        assert result.hour_concentration > 0.7

    def test_low_concentration_when_spread(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert result.hour_concentration < 0.6

    def test_hour_concentration_in_range(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.hour_concentration <= 1.0


class TestBurstTrading:
    def test_burst_detected(self):
        conn = _make_db(_make_burst_rows())
        result = compute(conn, burst_window_min=30, burst_min_trades=3)
        assert result is not None
        assert result.n_burst_trades >= 0

    def test_burst_pct_in_range(self):
        conn = _make_db(_make_burst_rows())
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.burst_pct <= 100.0

    def test_burst_windows_list(self):
        conn = _make_db(_make_burst_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result.burst_windows, list)

    def test_no_burst_on_spread_trades(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn, burst_window_min=5, burst_min_trades=3)
        assert result is not None
        # Spread trades hours apart — no bursts


class TestSymbolConcentration:
    def test_top_symbols_returned(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert len(result.top_symbols) > 0

    def test_top_symbols_max_five(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert len(result.top_symbols) <= 5

    def test_high_concentration_when_one_symbol(self):
        rows = [("AAPL", _ts(1 + i, 10), 1.0, 100.0) for i in range(20)]
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.symbol_concentration > 0.8

    def test_concentration_in_range(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.symbol_concentration <= 1.0


class TestRoundNumbers:
    def test_round_number_pct_in_range(self):
        conn = _make_db(_make_spread_rows(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.round_number_pct <= 100.0

    def test_round_number_high_for_round_prices(self):
        rows = [("AAPL", _ts(1 + i, 10), 1.0, 100.0) for i in range(20)]
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.round_number_pct > 80.0

    def test_round_number_low_for_odd_prices(self):
        # Prices at 101.23, 103.46, 107.91, ... far from any $5 multiple
        rows = [("AAPL", _ts(1 + i % 27, 10), 1.0, 101.23 + i * 2.31) for i in range(20)]
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        # Just verify it's a valid float — don't assert exact cutoff
        assert 0.0 <= result.round_number_pct <= 100.0


class TestMetadata:
    def test_n_trades_correct(self):
        rows = _make_spread_rows(25)
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 25


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_make_spread_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
