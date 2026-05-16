"""Tests for amms.analysis.symbol_performance."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.symbol_performance import SymbolStats, SymbolPerformanceReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (symbol, pnl)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (symbol, pnl) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, ?, '2026-01-01T09:00:00', '2026-01-02T14:00:00', 100.0, 105.0, 10.0, ?)",
            (i, symbol, pnl),
        )
    conn.commit()
    return conn


def _rows_mixed() -> list[tuple]:
    """AAPL profitable, TSLA marginal, NVDA losing."""
    rows = []
    for _ in range(5):
        rows.append(("AAPL", 100.0))
    rows.append(("AAPL", -20.0))
    for _ in range(3):
        rows.append(("TSLA", 50.0))
    for _ in range(3):
        rows.append(("TSLA", -40.0))
    for _ in range(2):
        rows.append(("NVDA", 30.0))
    for _ in range(4):
        rows.append(("NVDA", -80.0))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [("AAPL", 50.0), ("TSLA", -30.0)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, SymbolPerformanceReport)


class TestRanking:
    def test_best_symbol_correct(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.best_symbol == "AAPL"

    def test_worst_symbol_correct(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.worst_symbol == "NVDA"

    def test_sorted_by_total_pnl(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        pnls = [s.total_pnl for s in result.symbols]
        assert pnls == sorted(pnls, reverse=True)

    def test_ranks_assigned(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        ranks = [s.rank for s in result.symbols]
        assert ranks == list(range(1, len(result.symbols) + 1))

    def test_most_traded(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.most_traded == "AAPL"  # 6 trades


class TestStats:
    def test_win_rate_range(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        for s in result.symbols:
            assert 0 <= s.win_rate <= 100

    def test_n_trades_correct(self):
        rows = [("AAPL", 100.0)] * 4 + [("AAPL", -50.0)] * 2
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        aapl = next(s for s in result.symbols if s.symbol == "AAPL")
        assert aapl.n_trades == 6
        assert aapl.n_wins == 4
        assert aapl.n_losses == 2

    def test_total_pnl_correct(self):
        rows = [("AAPL", 100.0)] * 3 + [("AAPL", -50.0)] * 2
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        aapl = next(s for s in result.symbols if s.symbol == "AAPL")
        assert aapl.total_pnl == pytest.approx(200.0, abs=0.01)

    def test_profit_factor_positive(self):
        rows = [("AAPL", 100.0)] * 3 + [("AAPL", -50.0)] * 2
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        aapl = next(s for s in result.symbols if s.symbol == "AAPL")
        assert aapl.profit_factor is not None
        assert aapl.profit_factor > 1.0

    def test_min_trades_filter(self):
        rows = [("AAPL", 100.0)] * 5 + [("TSLA", 50.0)]  # TSLA only 1 trade
        conn = _make_conn(rows)
        result = compute(conn, min_trades=2)
        assert result is not None
        syms = [s.symbol for s in result.symbols]
        assert "AAPL" in syms
        assert "TSLA" not in syms

    def test_n_trades_sum(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.n_trades == sum(s.n_trades for s in result.symbols)

    def test_total_pnl_sum(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.total_pnl == pytest.approx(
            sum(s.total_pnl for s in result.symbols), abs=0.01
        )


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_n_symbols_correct(self):
        conn = _make_conn(_rows_mixed())
        result = compute(conn)
        assert result is not None
        assert result.n_symbols == 3
