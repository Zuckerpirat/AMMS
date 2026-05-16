"""Tests for amms.analysis.duration_return."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.duration_return import DurationReturnReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (buy_ts, sell_ts, pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (buy_ts, sell_ts, pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'SYM', ?, ?, ?, 101.0, ?, ?)",
            (i, buy_ts, sell_ts, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _trade(hold_days: float, pnl: float, buy_price: float = 100.0, qty: float = 10.0):
    buy_ts = "2026-01-01T10:00:00"
    sell_h = int(hold_days * 24)
    sell_ts = f"2026-01-{1 + hold_days // 1:02.0f}T{sell_h % 24:02d}:00:00"
    # Use a simple datetime offset
    from datetime import datetime, timedelta
    buy_dt = datetime(2026, 1, 1, 10, 0, 0)
    sell_dt = buy_dt + timedelta(days=hold_days)
    return (
        buy_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        sell_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        pnl, buy_price, qty,
    )


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [_trade(i, 50.0) for i in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        rows = [_trade(float(i), 50.0 if i % 2 == 0 else -30.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert isinstance(result, DurationReturnReport)


class TestBuckets:
    def test_default_four_buckets(self):
        rows = [_trade(float(i), 50.0 - i) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_buckets == 4

    def test_buckets_have_trades(self):
        rows = [_trade(float(i), 50.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        for b in result.buckets:
            assert b.n_trades > 0

    def test_bucket_days_ascending(self):
        rows = [_trade(float(i * 2), 50.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        min_days = [b.min_days for b in result.buckets]
        assert min_days == sorted(min_days)

    def test_win_rate_in_range(self):
        rows = [_trade(float(i), 50.0 if i % 2 == 0 else -30.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        for b in result.buckets:
            assert 0 <= b.win_rate <= 100

    def test_custom_n_buckets(self):
        rows = [_trade(float(i), 50.0) for i in range(24)]
        conn = _make_conn(rows)
        result = compute(conn, n_buckets=3)
        assert result is not None
        assert result.n_buckets == 3


class TestCorrelation:
    def test_positive_correlation_detected(self):
        """Longer holds → higher PnL → positive correlation."""
        rows = [_trade(float(i + 1), float(i * 10)) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.correlation is not None
        assert result.correlation > 0
        assert result.correlation_label == "positive"

    def test_negative_correlation_detected(self):
        """Longer holds → lower PnL → negative correlation."""
        rows = [_trade(float(i + 1), float((20 - i) * 10)) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.correlation is not None
        assert result.correlation < 0

    def test_correlation_bounded(self):
        rows = [_trade(float(i + 1), float(i * 5 + 10)) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        if result.correlation is not None:
            assert -1.0 <= result.correlation <= 1.0

    def test_correlation_label_valid(self):
        rows = [_trade(float(i), 50.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.correlation_label in ("positive", "negative", "none")


class TestOptimalAndVerdict:
    def test_optimal_bucket_present(self):
        rows = [_trade(float(i), 50.0 if i < 10 else -10.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.optimal_bucket is not None

    def test_verdict_present(self):
        rows = [_trade(float(i), 50.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_n_trades_correct(self):
        rows = [_trade(float(i), 50.0) for i in range(20)]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 20

    def test_limit_respected(self):
        rows = [_trade(float(i), 50.0) for i in range(50)]
        conn = _make_conn(rows)
        result = compute(conn, limit=20)
        assert result is not None
        assert result.n_trades == 20
