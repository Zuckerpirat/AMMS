"""Tests for amms.analysis.trade_quality."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.trade_quality import (
    TradeQualityScore,
    TradeQualityReport,
    compute_quality,
)


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    """Create in-memory DB with trade_pairs table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE trade_pairs (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            buy_ts TEXT,
            sell_ts TEXT,
            buy_price REAL,
            sell_price REAL,
            qty REAL,
            pnl REAL
        )"""
    )
    conn.executemany(
        "INSERT INTO trade_pairs (id, symbol, buy_ts, sell_ts, buy_price, sell_price, qty, pnl) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def _trade(
    trade_id=1,
    symbol="AAPL",
    buy_ts="2026-01-01T10:00:00",
    sell_ts="2026-01-05T10:00:00",
    buy_price=100.0,
    sell_price=110.0,
    qty=10.0,
    pnl=100.0,
):
    return (trade_id, symbol, buy_ts, sell_ts, buy_price, sell_price, qty, pnl)


class TestComputeQuality:
    def test_returns_none_empty_table(self):
        conn = _make_conn([])
        assert compute_quality(conn) is None

    def test_returns_none_no_completed_trades(self):
        """Trades without sell_price or pnl are excluded."""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE trade_pairs (id INTEGER, symbol TEXT, buy_ts TEXT, "
            "sell_ts TEXT, buy_price REAL, sell_price REAL, qty REAL, pnl REAL)"
        )
        conn.execute(
            "INSERT INTO trade_pairs VALUES (1, 'AAPL', '2026-01-01', NULL, 100, NULL, 10, NULL)"
        )
        conn.commit()
        assert compute_quality(conn) is None

    def test_returns_none_bad_table(self):
        """Missing table returns None gracefully."""
        conn = sqlite3.connect(":memory:")
        assert compute_quality(conn) is None

    def test_single_win_trade(self):
        conn = _make_conn([_trade(pnl=100.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.n_trades == 1
        assert result.scores[0].outcome_score == 30.0

    def test_single_loss_small(self):
        """Loss ≤ 2% → outcome_score = 15."""
        # buy_price=100, qty=10 → entry_value=1000, pnl=-15 → pnl_pct=-1.5%
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=-15.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].outcome_score == 15.0

    def test_single_loss_moderate(self):
        """Loss 2-5% → outcome_score = 5."""
        # entry_value=1000, pnl=-30 → pnl_pct=-3%
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=-30.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].outcome_score == 5.0

    def test_single_loss_large(self):
        """Loss > 5% → outcome_score = 0."""
        # entry_value=1000, pnl=-60 → pnl_pct=-6%
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=-60.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].outcome_score == 0.0

    def test_hold_score_swing(self):
        """Hold 1-5 days → hold_score = 20."""
        conn = _make_conn([_trade(
            buy_ts="2026-01-01T10:00:00",
            sell_ts="2026-01-03T10:00:00",  # 2 days
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_score == 20.0

    def test_hold_score_day_trade(self):
        """Hold < 1 day → hold_score = 5."""
        conn = _make_conn([_trade(
            buy_ts="2026-01-01T09:30:00",
            sell_ts="2026-01-01T15:30:00",  # 6 hours
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_score == 5.0

    def test_hold_score_medium(self):
        """Hold 5-30 days → hold_score = 15."""
        conn = _make_conn([_trade(
            buy_ts="2026-01-01T10:00:00",
            sell_ts="2026-01-15T10:00:00",  # 14 days
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_score == 15.0

    def test_hold_score_long(self):
        """Hold > 30 days → hold_score = 10."""
        conn = _make_conn([_trade(
            buy_ts="2026-01-01T10:00:00",
            sell_ts="2026-03-15T10:00:00",  # ~73 days
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_score == 10.0

    def test_hold_score_neutral_when_no_timestamps(self):
        """No timestamps → hold_score = 10 (neutral)."""
        conn = _make_conn([_trade(buy_ts=None, sell_ts=None)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_score == 10.0
        assert result.scores[0].hold_days is None

    def test_rr_score_high(self):
        """RR ≥ 2 → rr_score = 30."""
        # entry_value=1000, assumed_risk=10 (1%), pnl=25 → RR=2.5
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=25.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].rr_score == 30.0

    def test_rr_score_medium(self):
        """RR 1-2 → rr_score = 20."""
        # entry_value=1000, assumed_risk=10, pnl=15 → RR=1.5
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=15.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].rr_score == 20.0

    def test_grade_a(self):
        """Total ≥ 70 → grade A."""
        # Win trade (30), swing hold 3d (20), RR≥2 (30) = 80 total
        conn = _make_conn([_trade(
            buy_price=100.0, qty=10.0, pnl=25.0,
            buy_ts="2026-01-01T10:00:00",
            sell_ts="2026-01-04T10:00:00",  # 3 days swing
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].grade == "A"

    def test_grade_f(self):
        """Total < 25 → grade F."""
        # Large loss (0), day trade (5), bad RR → likely F
        conn = _make_conn([_trade(
            buy_price=100.0, qty=10.0, pnl=-100.0,
            buy_ts="2026-01-01T09:30:00",
            sell_ts="2026-01-01T14:00:00",
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].grade == "F"

    def test_report_avg_score(self):
        trades = [
            _trade(trade_id=1, pnl=100.0),
            _trade(trade_id=2, pnl=-60.0, buy_price=100.0, qty=10.0),
        ]
        conn = _make_conn(trades)
        result = compute_quality(conn)
        assert result is not None
        assert result.n_trades == 2
        assert isinstance(result.avg_score, float)

    def test_report_grade_distribution(self):
        trades = [_trade(trade_id=i, pnl=100.0) for i in range(5)]
        conn = _make_conn(trades)
        result = compute_quality(conn)
        assert result is not None
        total_graded = sum(result.grade_distribution.values())
        assert total_graded == 5

    def test_report_best_worst(self):
        trades = [
            _trade(trade_id=1, pnl=100.0,
                   buy_ts="2026-01-01T10:00:00",
                   sell_ts="2026-01-04T10:00:00"),
            _trade(trade_id=2, pnl=-100.0, buy_price=100.0, qty=10.0,
                   buy_ts="2026-01-01T09:30:00",
                   sell_ts="2026-01-01T14:00:00"),
        ]
        conn = _make_conn(trades)
        result = compute_quality(conn)
        assert result is not None
        assert result.best_trade is not None
        assert result.worst_quality_trade is not None
        assert result.best_trade.total_score >= result.worst_quality_trade.total_score

    def test_limit_respected(self):
        trades = [_trade(trade_id=i, pnl=10.0) for i in range(20)]
        conn = _make_conn(trades)
        result = compute_quality(conn, limit=5)
        assert result is not None
        assert result.n_trades == 5

    def test_pnl_pct_computed(self):
        """PnL% = pnl / (buy_price * qty) * 100."""
        conn = _make_conn([_trade(buy_price=100.0, qty=10.0, pnl=100.0)])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].pnl_pct == pytest.approx(10.0, abs=0.01)

    def test_hold_days_computed(self):
        conn = _make_conn([_trade(
            buy_ts="2026-01-01T10:00:00",
            sell_ts="2026-01-04T10:00:00",
        )])
        result = compute_quality(conn)
        assert result is not None
        assert result.scores[0].hold_days == pytest.approx(3.0, abs=0.01)

    def test_scores_list_length(self):
        trades = [_trade(trade_id=i, pnl=5.0 * i) for i in range(1, 8)]
        conn = _make_conn(trades)
        result = compute_quality(conn)
        assert result is not None
        assert len(result.scores) == 7
