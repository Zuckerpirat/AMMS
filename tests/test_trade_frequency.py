"""Tests for amms.analysis.trade_frequency."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.trade_frequency import FrequencyBucket, TradeFrequencyReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (sell_ts, pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (sell_ts, pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', '2026-01-01T09:00:00', ?, ?, 105.0, ?, ?)",
            (i, sell_ts, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _make_rows_spread(n_days: int = 10, trades_per_day: int = 2) -> list[tuple]:
    """n_days trading days each with trades_per_day trades."""
    rows = []
    for day in range(1, n_days + 1):
        for trade in range(trades_per_day):
            hour = 9 + trade
            pnl = 50.0 if (day + trade) % 2 == 0 else -30.0
            rows.append((
                f"2026-01-{day:02d}T{hour:02d}:00:00",
                pnl, 100.0, 10.0,
            ))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few_trades(self):
        rows = [("2026-01-05T10:00:00", 50.0, 100.0, 10.0) for _ in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, TradeFrequencyReport)


class TestFrequencyStats:
    def test_avg_trades_per_day_correct(self):
        # 10 days × 2 trades = 20 trades, avg = 2/day
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.avg_trades_per_day == pytest.approx(2.0, abs=0.1)

    def test_avg_trades_per_week_approx(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.avg_trades_per_week == pytest.approx(10.0, abs=1.0)

    def test_n_trading_days_correct(self):
        # 10 distinct days
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.n_trading_days == 10

    def test_n_trades_correct(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20

    def test_most_active_day(self):
        rows = _make_rows_spread(5, 1)
        # Add a day with 5 trades
        for h in range(5):
            rows.append((f"2026-01-20T{9+h:02d}:00:00", 50.0, 100.0, 10.0))
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.most_active_day_count == 5


class TestBuckets:
    def test_buckets_present(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert len(result.buckets) > 0

    def test_win_days_pct_range(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        for b in result.buckets:
            assert 0 <= b.win_days_pct <= 100

    def test_best_worst_bucket_present(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.best_bucket is not None
        assert result.worst_bucket is not None


class TestCorrelation:
    def test_correlation_bounded(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        if result.correlation is not None:
            assert -1.0 <= result.correlation <= 1.0

    def test_correlation_label_valid(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert result.correlation_label in ("positive", "negative", "none")

    def test_negative_correlation_when_overtrading_hurts(self):
        """1 trade/day wins, 5+ trades/day loses → negative correlation."""
        rows = []
        for day in range(1, 8):
            rows.append((f"2026-01-{day:02d}T10:00:00", 200.0, 100.0, 10.0))
        for day in range(10, 17):
            for h in range(5):
                rows.append((f"2026-01-{day:02d}T{9+h:02d}:00:00", -50.0, 100.0, 10.0))
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.correlation is not None
        assert result.correlation < 0


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_make_rows_spread(10, 2))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
