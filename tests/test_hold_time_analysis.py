"""Tests for amms.analysis.hold_time_analysis."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.hold_time_analysis import (
    BucketStats, HoldTimeReport, compute, _classify,
)


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER, symbol TEXT, buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    conn.executemany(
        "INSERT INTO trade_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    return conn


def _trade(tid, buy_ts, sell_ts, buy_price=100.0, qty=10.0, pnl=50.0):
    return (tid, "SYM", buy_ts, sell_ts, buy_price, buy_price + 0.5, qty, pnl)


class TestClassify:
    def test_day(self):
        assert _classify(0.5) == "day"
        assert _classify(1.9) == "day"

    def test_swing(self):
        assert _classify(2.0) == "swing"
        assert _classify(13.9) == "swing"

    def test_medium(self):
        assert _classify(14.0) == "medium"
        assert _classify(89.9) == "medium"

    def test_long(self):
        assert _classify(90.0) == "long"
        assert _classify(365.0) == "long"


class TestCompute:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_none_no_timestamps(self):
        """Trades without timestamps are skipped."""
        conn = _make_conn([_trade(1, None, None)])
        assert compute(conn) is None

    def test_single_day_trade(self):
        conn = _make_conn([
            _trade(1, "2026-01-01T09:30:00", "2026-01-01T15:30:00", pnl=50.0)
        ])
        result = compute(conn)
        assert result is not None
        day_bucket = next((b for b in result.buckets if b.bucket == "day"), None)
        assert day_bucket is not None
        assert day_bucket.n_trades == 1

    def test_single_swing_trade(self):
        conn = _make_conn([
            _trade(1, "2026-01-01T10:00:00", "2026-01-05T10:00:00", pnl=75.0)
        ])
        result = compute(conn)
        assert result is not None
        swing = next((b for b in result.buckets if b.bucket == "swing"), None)
        assert swing is not None

    def test_win_rate_all_wins(self):
        rows = [
            _trade(i, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0)
            for i in range(5)
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        swing = next((b for b in result.buckets if b.bucket == "swing"), None)
        assert swing is not None
        assert swing.win_rate == pytest.approx(100.0)

    def test_win_rate_mixed(self):
        rows = [
            _trade(1, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0),
            _trade(2, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=-50.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        swing = next((b for b in result.buckets if b.bucket == "swing"), None)
        assert swing is not None
        assert swing.win_rate == pytest.approx(50.0)

    def test_profit_factor_computed(self):
        rows = [
            _trade(1, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=100.0),
            _trade(2, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=-50.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        swing = next((b for b in result.buckets if b.bucket == "swing"), None)
        assert swing is not None
        assert swing.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_profit_factor_none_when_no_losses(self):
        rows = [
            _trade(i, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0)
            for i in range(3)
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        swing = next(b for b in result.buckets if b.bucket == "swing")
        assert swing.profit_factor is None

    def test_multiple_buckets(self):
        rows = [
            _trade(1, "2026-01-01T09:30:00", "2026-01-01T15:30:00", pnl=10.0),  # day
            _trade(2, "2026-01-01T10:00:00", "2026-01-05T10:00:00", pnl=50.0),  # swing
            _trade(3, "2026-01-01T10:00:00", "2026-02-05T10:00:00", pnl=80.0),  # medium
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert len(result.buckets) == 3

    def test_best_bucket_identified(self):
        rows = [
            _trade(1, "2026-01-01T09:30:00", "2026-01-01T15:30:00", pnl=-50.0),
            _trade(2, "2026-01-01T10:00:00", "2026-01-05T10:00:00", pnl=80.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.best_bucket == "swing"

    def test_total_trades_correct(self):
        rows = [
            _trade(i, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=10.0)
            for i in range(7)
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.total_trades == 7

    def test_overall_win_rate(self):
        rows = [
            _trade(1, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0),
            _trade(2, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=-50.0),
            _trade(3, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0),
            _trade(4, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.overall_win_rate == pytest.approx(75.0, abs=0.1)

    def test_best_worst_pnl_in_bucket(self):
        rows = [
            _trade(1, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=200.0),
            _trade(2, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=-30.0),
            _trade(3, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=50.0),
        ]
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        swing = next(b for b in result.buckets if b.bucket == "swing")
        assert swing.best_pnl == pytest.approx(200.0, abs=0.1)
        assert swing.worst_pnl == pytest.approx(-30.0, abs=0.1)

    def test_limit_respected(self):
        rows = [
            _trade(i, "2026-01-01T10:00:00", "2026-01-04T10:00:00", pnl=10.0)
            for i in range(20)
        ]
        conn = _make_conn(rows)
        result = compute(conn, limit=5)
        assert result is not None
        assert result.total_trades == 5
