"""Tests for amms.analysis.size_performance."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.size_performance import SizeTier, SizePerformanceReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (buy_price, qty, pnl)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (buy_price, qty, pnl) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', '2026-01-02T14:00:00', ?, 105.0, ?, ?)",
            (i, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _small_large_rows() -> list[tuple]:
    """Small positions lose, large positions win."""
    rows = []
    # Small: $100 positions, losing
    for _ in range(5):
        rows.append((10.0, 10.0, -50.0))   # $100, -50% = -50
    # Medium: $500
    for _ in range(5):
        rows.append((50.0, 10.0, 10.0))    # $500, +2%
    # Large: $1000
    for _ in range(5):
        rows.append((100.0, 10.0, 80.0))   # $1000, +8%
    # XLarge: $5000
    for _ in range(5):
        rows.append((500.0, 10.0, 400.0))  # $5000, +8%
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(100.0, 10.0, 50.0) for _ in range(5)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, SizePerformanceReport)


class TestTiers:
    def test_four_tiers_default(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.tiers) == 4

    def test_tier_labels(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        labels = [t.label for t in result.tiers]
        assert "Small" in labels
        assert "Large" in labels

    def test_tier_sizes_ascending(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        min_vals = [t.min_value for t in result.tiers]
        assert min_vals == sorted(min_vals)

    def test_win_rate_range(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        for t in result.tiers:
            assert 0 <= t.win_rate <= 100

    def test_n_trades_sum(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert result.n_trades == sum(t.n_trades for t in result.tiers)


class TestCorrelation:
    def test_positive_correlation(self):
        """Large positions → higher PnL% → positive correlation."""
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert result.correlation is not None
        assert result.correlation > 0
        assert result.correlation_label == "positive"

    def test_negative_correlation(self):
        """Small positions win big, large positions lose → negative."""
        rows = []
        for _ in range(5):
            rows.append((10.0, 10.0, 80.0))   # $100, +80%
        for _ in range(5):
            rows.append((50.0, 10.0, 20.0))   # $500, +4%
        for _ in range(5):
            rows.append((100.0, 10.0, -50.0)) # $1000, -5%
        for _ in range(5):
            rows.append((500.0, 10.0, -400.0))# $5000, -8%
        conn = _make_conn(rows)
        result = compute(conn)
        assert result is not None
        assert result.correlation is not None
        assert result.correlation < 0
        assert result.correlation_label == "negative"

    def test_correlation_bounded(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        if result.correlation is not None:
            assert -1.0 <= result.correlation <= 1.0

    def test_correlation_label_valid(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert result.correlation_label in ("positive", "negative", "none")


class TestBestWorst:
    def test_best_tier_present(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert result.best_tier is not None

    def test_worst_tier_present(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert result.worst_tier is not None

    def test_best_is_xlarge_when_large_wins(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        best = next(t for t in result.tiers if t.label == result.best_tier)
        worst = next(t for t in result.tiers if t.label == result.worst_tier)
        assert best.avg_pnl_pct >= worst.avg_pnl_pct


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_small_large_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
