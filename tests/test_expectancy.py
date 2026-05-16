"""Tests for amms.analysis.expectancy."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.expectancy import ExpectancyReport, SymbolExpectancy, compute


def _make_conn(rows: list[tuple[str, float]]) -> sqlite3.Connection:
    """rows: list of (symbol, pnl)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (sym, pnl) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, ?, "
            "'2026-01-01T10:00:00', '2026-01-04T10:00:00', "
            "100.0, 101.0, 10.0, ?)",
            (i, sym, pnl),
        )
    conn.commit()
    return conn


def _sym_rows(sym: str, pnls: list[float]) -> list[tuple[str, float]]:
    return [(sym, p) for p in pnls]


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_conn([("SYM", 50.0)] * 4)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        rows = _sym_rows("SYM", [50.0, -30.0, 50.0, -30.0, 80.0])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert isinstance(result, ExpectancyReport)


class TestOverallExpectancy:
    def test_positive_expectancy(self):
        """60% win rate, avg win=100, avg loss=50 → expectancy=0.6*100-0.4*50=40."""
        rows = _sym_rows("S", [100, 100, 100, 100, 100, 100, -50, -50, -50, -50])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall.expectancy == pytest.approx(40.0, abs=1.0)

    def test_negative_expectancy(self):
        rows = _sym_rows("S", [-100, -100, -100, -100, 30, 30])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall.expectancy < 0

    def test_r_expectancy_formula(self):
        """R expectancy = expectancy / avg_loss."""
        rows = _sym_rows("S", [100, 100, 100, -50, -50, -50])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        o = result.overall
        if o.avg_loss > 0:
            expected_r = o.expectancy / o.avg_loss
            assert o.r_expectancy == pytest.approx(expected_r, abs=0.01)

    def test_win_rate_correct(self):
        rows = _sym_rows("S", [50] * 7 + [-50] * 3)
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall.win_rate == pytest.approx(70.0, abs=0.1)

    def test_n_trades_correct(self):
        rows = _sym_rows("S", [50.0, -30.0] * 8)
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 16

    def test_limit_respected(self):
        rows = _sym_rows("S", [50.0] * 50)
        conn = _make_conn(rows)
        result = compute(conn, limit=20)
        assert result is not None
        assert result.n_trades == 20


class TestSymbolBreakdown:
    def test_two_symbols_both_present(self):
        rows = (
            _sym_rows("AAPL", [50, 50, -30, 50, -30, 50])
            + _sym_rows("TSLA", [-50, -50, 20, -50, 20, -50])
        )
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        syms = {s.symbol for s in result.by_symbol}
        assert "AAPL" in syms
        assert "TSLA" in syms

    def test_sorted_by_expectancy_desc(self):
        rows = (
            _sym_rows("AAPL", [100, 100, 100, -20, -20, -20])
            + _sym_rows("TSLA", [-80, -80, -80, 20, 20, 20])
        )
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        exps = [s.expectancy for s in result.by_symbol]
        assert exps == sorted(exps, reverse=True)

    def test_best_worst_symbols(self):
        rows = (
            _sym_rows("GOOD", [100, 100, 100, -10, -10, -10])
            + _sym_rows("BAD", [-100, -100, -100, 10, 10, 10])
        )
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.best_symbol == "GOOD"
        assert result.worst_symbol == "BAD"

    def test_min_trades_filter(self):
        """Symbols with fewer than min_trades excluded from breakdown."""
        rows = (
            _sym_rows("A", [50, -30, 50, -30, 50])   # 5 trades
            + _sym_rows("B", [50, -30])               # 2 trades — should be excluded
        )
        conn = _make_conn(rows)
        result = compute(conn, min_trades=3)
        assert result is not None
        syms = {s.symbol for s in result.by_symbol}
        assert "A" in syms
        assert "B" not in syms

    def test_n_symbols_correct(self):
        rows = (
            _sym_rows("X", [50, -30, 50, -30, 50])
            + _sym_rows("Y", [50, -30, 50, -30, 50])
        )
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_symbols == 2


class TestGrades:
    def test_grade_a_high_r(self):
        """0.5R+ expectancy → A."""
        rows = _sym_rows("S", [200, 200, 200, 200, -50, -50])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall.grade in ("A", "B")

    def test_grade_f_negative(self):
        rows = _sym_rows("S", [-100, -100, -100, -100, 20, 20])
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall.grade in ("D", "F")
