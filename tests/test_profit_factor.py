"""Tests for amms.analysis.profit_factor."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.profit_factor import ProfitFactorReport, RollingPF, SymbolPF, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl_pct, symbol)"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trades (pnl_pct REAL, symbol TEXT, closed_at TEXT, status TEXT)")
    for i, (pnl, sym) in enumerate(rows):
        ts = f"2024-01-{1 + i % 28:02d} {i % 24:02d}:00:00"
        conn.execute("INSERT INTO trades VALUES (?, ?, ?, 'closed')", (pnl, sym, ts))
    conn.commit()
    return conn


def _good_trades(n: int = 30) -> list[tuple]:
    """2:1 win/loss ratio, 60% WR."""
    pattern = [(2.0, "AAPL"), (2.0, "AAPL"), (-1.0, "AAPL")]
    return (pattern * (n // 3 + 1))[:n]


def _bad_trades(n: int = 30) -> list[tuple]:
    """0.5:1 win/loss ratio, 40% WR."""
    pattern = [(0.5, "JPM"), (-1.0, "JPM"), (-1.0, "JPM")]
    return (pattern * (n // 3 + 1))[:n]


def _mixed_symbols(n: int = 30) -> list[tuple]:
    syms = ["AAPL", "TSLA", "MSFT"]
    pnls = [2.0, -1.0, 1.5, -0.8, 0.9, -1.2]
    return [(pnls[i % 6], syms[i % 3]) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db(_good_trades(5))
        assert compute(conn) is None

    def test_returns_none_all_wins(self):
        conn = _make_db([(1.0, "X")] * 15)
        assert compute(conn) is None

    def test_returns_none_all_losses(self):
        conn = _make_db([(-1.0, "X")] * 15)
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_db(_good_trades(20))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, ProfitFactorReport)


class TestProfitFactor:
    def test_pf_positive(self):
        conn = _make_db(_good_trades(20))
        result = compute(conn)
        assert result is not None
        assert result.profit_factor > 0

    def test_good_trades_pf_above_one(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.profit_factor > 1.0

    def test_bad_trades_pf_below_one(self):
        conn = _make_db(_bad_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.profit_factor < 1.0

    def test_pf_equals_gross_ratio(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        expected = result.gross_wins / result.gross_losses if result.gross_losses > 0 else 999.0
        assert abs(result.profit_factor - expected) < 0.001


class TestWinRate:
    def test_win_rate_in_range(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.win_rate <= 100.0

    def test_win_rate_matches_pattern(self):
        # _good_trades: 2 wins per 3 trades = 66.7%
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert abs(result.win_rate - 66.7) < 1.0

    def test_n_winners_n_losers_sum(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.n_winners + result.n_losers == result.n_trades


class TestPayoff:
    def test_payoff_positive(self):
        conn = _make_db(_good_trades(20))
        result = compute(conn)
        assert result is not None
        assert result.payoff_ratio > 0

    def test_payoff_ratio_above_one_for_good_trades(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.payoff_ratio > 1.0


class TestBreakeven:
    def test_breakeven_in_range(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.breakeven_win_rate <= 100.0

    def test_edge_positive_for_good_trades(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.edge > 0

    def test_edge_negative_for_bad_trades(self):
        conn = _make_db(_bad_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.edge < 0


class TestKelly:
    def test_kelly_nonneg(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.kelly_pct >= 0.0

    def test_kelly_positive_for_good_trades(self):
        conn = _make_db(_good_trades(30))
        result = compute(conn)
        assert result is not None
        assert result.kelly_pct > 0


class TestRolling:
    def test_rolling_present_for_large_dataset(self):
        conn = _make_db(_good_trades(40))
        result = compute(conn)
        assert result is not None
        assert result.rolling_20 is not None
        assert isinstance(result.rolling_20, RollingPF)

    def test_rolling_trend_is_valid(self):
        conn = _make_db(_good_trades(40))
        result = compute(conn)
        assert result is not None
        assert result.rolling_pf_trend in {"improving", "deteriorating", "stable"}


class TestBySymbol:
    def test_by_symbol_not_empty(self):
        conn = _make_db(_mixed_symbols(30))
        result = compute(conn)
        assert result is not None
        assert len(result.by_symbol) > 0

    def test_by_symbol_max_five(self):
        conn = _make_db(_mixed_symbols(30))
        result = compute(conn)
        assert result is not None
        assert len(result.by_symbol) <= 5

    def test_by_symbol_is_symbolpf(self):
        conn = _make_db(_mixed_symbols(30))
        result = compute(conn)
        assert result is not None
        for s in result.by_symbol:
            assert isinstance(s, SymbolPF)


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_good_trades(20))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_profit_factor(self):
        conn = _make_db(_good_trades(20))
        result = compute(conn)
        assert result is not None
        assert "profit" in result.verdict.lower() or "factor" in result.verdict.lower()
