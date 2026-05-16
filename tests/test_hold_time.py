"""Tests for amms.analysis.hold_time."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.hold_time import HoldBucket, HoldTimeReport, SymbolHoldStats, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl_pct, entered_at, closed_at, symbol)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trades (pnl_pct REAL, entered_at TEXT, closed_at TEXT, symbol TEXT, status TEXT)"
    )
    conn.executemany("INSERT INTO trades VALUES (?, ?, ?, ?, 'closed')", rows)
    conn.commit()
    return conn


def _scalp_rows(n: int = 15) -> list[tuple]:
    """All intraday scalps (10-minute trades)."""
    rows = []
    for i in range(n):
        entered = f"2024-01-{1 + i % 28:02d} 10:00:00"
        closed = f"2024-01-{1 + i % 28:02d} 10:10:00"
        pnl = 1.0 if i % 3 != 0 else -0.5
        rows.append((pnl, entered, closed, "AAPL"))
    return rows


def _swing_rows(n: int = 15) -> list[tuple]:
    """Swing trades held 2 days."""
    rows = []
    for i in range(n):
        day = 1 + i * 2
        if day > 28:
            day = 1
        entered = f"2024-01-{day:02d} 10:00:00"
        closed = f"2024-01-{min(day + 2, 28):02d} 10:00:00"
        pnl = 1.5 if i % 2 == 0 else -0.8
        sym = "TSLA" if i % 2 == 0 else "MSFT"
        rows.append((pnl, entered, closed, sym))
    return rows


def _mixed_rows(n: int = 20) -> list[tuple]:
    """Mix of scalp and swing trades."""
    rows = []
    for i in range(n):
        if i % 3 == 0:
            entered = f"2024-01-{1 + i % 28:02d} 09:30:00"
            closed = f"2024-01-{1 + i % 28:02d} 09:45:00"
        else:
            entered = f"2024-01-{1 + i % 14:02d} 10:00:00"
            closed = f"2024-01-{(1 + i % 14 + 3) % 28 + 1:02d} 10:00:00"
        pnl = 1.0 if i % 2 == 0 else -0.5
        rows.append((pnl, entered, closed, f"SYM{i % 4}"))
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        conn = _make_db(_scalp_rows(2))
        assert compute(conn) is None

    def test_returns_result_enough_trades(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        assert isinstance(result, HoldTimeReport)


class TestHoldMetrics:
    def test_median_hold_positive(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        assert result.median_hold_min > 0

    def test_mean_hold_positive(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        assert result.mean_hold_min > 0

    def test_min_le_median_le_max(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.min_hold_min <= result.median_hold_min
        assert result.median_hold_min <= result.max_hold_min

    def test_scalp_median_is_short(self):
        conn = _make_db(_scalp_rows(15))
        result = compute(conn)
        assert result is not None
        assert result.median_hold_min <= 30  # scalp = < 30 min


class TestBuckets:
    def test_four_buckets_returned(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.buckets) == 4

    def test_bucket_labels(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        labels = [b.label for b in result.buckets]
        assert "scalp" in labels
        assert "intraday" in labels
        assert "swing" in labels
        assert "multi-day" in labels

    def test_bucket_is_dataclass(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        for b in result.buckets:
            assert isinstance(b, HoldBucket)

    def test_scalp_bucket_populated_for_scalp_trades(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        scalp_bucket = next(b for b in result.buckets if b.label == "scalp")
        assert scalp_bucket.n_trades > 0

    def test_bucket_win_rate_in_range(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        for b in result.buckets:
            if b.n_trades > 0:
                assert 0.0 <= b.win_rate <= 100.0


class TestBestWorstBucket:
    def test_best_bucket_present(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.best_bucket is not None

    def test_best_ge_worst(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        if result.best_bucket and result.worst_bucket:
            assert result.best_bucket.avg_pnl_pct >= result.worst_bucket.avg_pnl_pct - 0.01


class TestCorrelation:
    def test_correlation_in_range(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert -1.0 <= result.hold_pnl_correlation <= 1.0


class TestBySymbol:
    def test_by_symbol_list_not_empty(self):
        conn = _make_db(_swing_rows(15))
        result = compute(conn)
        assert result is not None
        assert len(result.by_symbol) > 0

    def test_by_symbol_is_stats(self):
        conn = _make_db(_swing_rows(15))
        result = compute(conn)
        assert result is not None
        for s in result.by_symbol:
            assert isinstance(s, SymbolHoldStats)

    def test_by_symbol_max_five(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert len(result.by_symbol) <= 5


class TestHoldTrend:
    def test_hold_trend_is_valid(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.hold_trend in {"increasing", "decreasing", "stable"}


class TestNTrades:
    def test_n_trades_correct(self):
        rows = _scalp_rows(12)
        conn = _make_db(rows)
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 12

    def test_n_with_duration_le_n_trades(self):
        conn = _make_db(_mixed_rows(20))
        result = compute(conn)
        assert result is not None
        assert result.n_with_duration <= result.n_trades


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_hold(self):
        conn = _make_db(_scalp_rows(10))
        result = compute(conn)
        assert result is not None
        assert "hold" in result.verdict.lower()
