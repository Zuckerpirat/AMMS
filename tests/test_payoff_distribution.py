"""Tests for amms.analysis.payoff_distribution."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.payoff_distribution import ReturnBucket, PayoffDistributionReport, compute


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pnl, buy_price, qty)"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trade_pairs ("
        "id INTEGER PRIMARY KEY, symbol TEXT, "
        "buy_ts TEXT, sell_ts TEXT, "
        "buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
    )
    for i, (pnl, buy_price, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_pairs VALUES (?, 'AAPL', '2026-01-01T09:00:00', '2026-01-02T14:00:00', ?, 105.0, ?, ?)",
            (i, buy_price, qty, pnl),
        )
    conn.commit()
    return conn


def _balanced_rows(n: int = 30) -> list[tuple]:
    """Mixed returns: small wins and losses, some big ones."""
    rows = []
    for i in range(n):
        if i % 6 == 0:
            pnl = 700.0   # big win ~7%
        elif i % 6 == 1:
            pnl = 300.0   # win ~3%
        elif i % 6 == 2:
            pnl = 100.0   # small win ~1%
        elif i % 6 == 3:
            pnl = -100.0  # small loss ~-1%
        elif i % 6 == 4:
            pnl = -300.0  # loss ~-3%
        else:
            pnl = -700.0  # big loss ~-7%
        rows.append((pnl, 100.0, 100.0))  # entry value $10000
    return rows


def _right_skewed_rows(n: int = 30) -> list[tuple]:
    """Many small losses, few huge wins."""
    rows = []
    for i in range(n):
        if i < 5:
            rows.append((5000.0, 100.0, 100.0))   # huge win ~50%
        else:
            rows.append((-100.0, 100.0, 100.0))   # small loss ~-1%
    return rows


class TestEdgeCases:
    def test_returns_none_empty(self):
        conn = _make_conn([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(50.0, 100.0, 10.0) for _ in range(10)]
        conn = _make_conn(rows)
        assert compute(conn) is None

    def test_returns_none_bad_table(self):
        conn = sqlite3.connect(":memory:")
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, PayoffDistributionReport)


class TestBuckets:
    def test_six_buckets(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.buckets) == 6

    def test_bucket_pcts_sum_to_100(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        total = sum(b.pct_of_total for b in result.buckets)
        assert total == pytest.approx(100.0, abs=1.0)

    def test_n_trades_sum(self):
        conn = _make_conn(_balanced_rows(30))
        result = compute(conn)
        assert result is not None
        assert result.n_trades == 30
        assert sum(b.n_trades for b in result.buckets) == 30

    def test_big_win_bucket_populated(self):
        conn = _make_conn(_right_skewed_rows())
        result = compute(conn)
        assert result is not None
        big_win = next(b for b in result.buckets if "Big Win" in b.label)
        assert big_win.n_trades > 0

    def test_big_loss_bucket_populated(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        big_loss = next(b for b in result.buckets if "Big Loss" in b.label)
        assert big_loss.n_trades > 0


class TestDistributionStats:
    def test_skewness_finite(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert abs(result.skewness) < 1000

    def test_right_skewed_positive_skew(self):
        conn = _make_conn(_right_skewed_rows())
        result = compute(conn)
        assert result is not None
        assert result.skewness > 0
        assert result.distribution_shape == "right_skewed"

    def test_shape_valid(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.distribution_shape in ("right_skewed", "left_skewed", "normal", "fat_tails")

    def test_mean_median_present(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result.mean_return, float)
        assert isinstance(result.median_return, float)

    def test_top10_contribution_range(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        # top10 contribution can be any float (could be negative if top 10% of trades still lose)
        assert isinstance(result.top10_pct_contribution, float)


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_conn(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10
