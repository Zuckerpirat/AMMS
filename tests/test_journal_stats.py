"""Tests for extended journal statistics."""

from __future__ import annotations

import sqlite3

from amms.analysis.journal_stats import JournalStats, compute


def _make_conn(trades: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE trade_pairs (
            id INTEGER PRIMARY KEY, symbol TEXT, buy_ts TEXT, sell_ts TEXT,
            buy_price REAL, sell_price REAL, qty REAL, pnl REAL
        )
    """)
    for i, (pnl, buy_ts, sell_ts) in enumerate(trades):
        conn.execute(
            "INSERT INTO trade_pairs (symbol, buy_ts, sell_ts, buy_price, sell_price, qty, pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", buy_ts, sell_ts, 100.0, 100.0 + pnl / 10, 10.0, pnl)
        )
    conn.commit()
    return conn


GOOD_TRADES = [
    (150.0, "2026-01-01", "2026-01-05"),
    (200.0, "2026-01-06", "2026-01-12"),
    (-50.0, "2026-01-13", "2026-01-15"),
    (100.0, "2026-01-16", "2026-01-20"),
    (-75.0, "2026-01-21", "2026-01-23"),
]


class TestJournalStats:
    def test_returns_stats(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert isinstance(result, JournalStats)

    def test_n_trades_correct(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 5

    def test_win_rate_correct(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert abs(result.win_rate - 0.6) < 0.01  # 3 wins, 2 losses

    def test_total_pnl_correct(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        expected = 150 + 200 - 50 + 100 - 75
        assert abs(result.total_pnl - expected) < 0.01

    def test_profit_factor_positive(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert result.profit_factor > 1.0

    def test_expectancy_positive_for_winning_strategy(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert result.expectancy > 0

    def test_empty_table_returns_none(self) -> None:
        conn = _make_conn([])
        result = compute(conn)
        assert result is None

    def test_max_streaks_computed(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert result.max_win_streak >= 1
        assert result.max_loss_streak >= 1

    def test_largest_win_loss(self) -> None:
        conn = _make_conn(GOOD_TRADES)
        result = compute(conn)
        assert result is not None
        assert result.largest_win == 200.0
        assert result.largest_loss == 75.0

    def test_all_losing_strategy(self) -> None:
        trades = [(-100.0, "2026-01-01", "2026-01-02")] * 5
        conn = _make_conn(trades)
        result = compute(conn)
        assert result is not None
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
