"""Tests for amms.analysis.position_aging."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from amms.analysis.position_aging import AgingReport, PositionAge, analyze_aging


def _make_position(sym: str, entry_price: float = 100.0, market_value: float = 105.0, qty: float = 1.0):
    class P:
        pass
    p = P()
    p.symbol = sym
    p.avg_entry_price = entry_price
    p.market_value = market_value
    p.qty = qty
    return p


class _FakeBroker:
    def __init__(self, positions=None):
        self._positions = positions or [_make_position("AAPL")]

    def get_positions(self):
        return self._positions


class _EmptyBroker:
    def get_positions(self):
        return []


class _ErrorBroker:
    def get_positions(self):
        raise RuntimeError("error")


def _make_db(buy_dates: dict[str, str]) -> sqlite3.Connection:
    """Create in-memory DB with trade_pairs table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE trade_pairs (
            id INTEGER PRIMARY KEY, symbol TEXT, buy_ts TEXT, sell_ts TEXT,
            buy_price REAL, sell_price REAL, qty REAL, pnl REAL
        )
    """)
    for sym, buy_ts in buy_dates.items():
        conn.execute(
            "INSERT INTO trade_pairs (symbol, buy_ts, sell_ts, buy_price, sell_price, qty, pnl) "
            "VALUES (?,?,?,?,?,?,?)",
            (sym, buy_ts, None, 100.0, None, 1.0, None)
        )
    conn.commit()
    return conn


class TestAnalyzeAging:
    def test_returns_none_no_positions(self):
        assert analyze_aging(_EmptyBroker()) is None

    def test_returns_none_broker_error(self):
        assert analyze_aging(_ErrorBroker()) is None

    def test_returns_report_with_position(self):
        result = analyze_aging(_FakeBroker())
        assert result is not None
        assert isinstance(result, AgingReport)

    def test_total_positions_correct(self):
        broker = _FakeBroker([_make_position("AAPL"), _make_position("MSFT")])
        result = analyze_aging(broker)
        assert result is not None
        assert result.total_positions == 2

    def test_unknown_hold_style_without_db(self):
        result = analyze_aging(_FakeBroker())
        assert result is not None
        assert result.positions[0].hold_style == "unknown"
        assert result.positions[0].hold_days is None

    def test_hold_days_computed_from_db(self):
        # Entry 10 days ago
        entry_date = (datetime.now(UTC).date() - timedelta(days=10)).isoformat()
        conn = _make_db({"AAPL": entry_date})
        result = analyze_aging(_FakeBroker(), conn=conn)
        assert result is not None
        pos = next(p for p in result.positions if p.symbol == "AAPL")
        assert pos.hold_days is not None
        assert 9 <= pos.hold_days <= 11  # tolerance for date calculation

    def test_swing_style_for_5_day_hold(self):
        entry_date = (datetime.now(UTC).date() - timedelta(days=5)).isoformat()
        conn = _make_db({"AAPL": entry_date})
        result = analyze_aging(_FakeBroker(), conn=conn)
        assert result is not None
        pos = result.positions[0]
        assert pos.hold_style == "swing"

    def test_medium_style_for_30_day_hold(self):
        entry_date = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
        conn = _make_db({"AAPL": entry_date})
        result = analyze_aging(_FakeBroker(), conn=conn)
        assert result is not None
        pos = result.positions[0]
        assert pos.hold_style == "medium"

    def test_pnl_pct_computed(self):
        # Entry at 100, current value = 110 (qty=1) → +10%
        pos = _make_position("AAPL", entry_price=100.0, market_value=110.0, qty=1.0)
        result = analyze_aging(_FakeBroker([pos]))
        assert result is not None
        assert result.positions[0].pnl_pct == pytest.approx(10.0, abs=0.1)

    def test_overstay_flag_for_long_losing_position(self):
        # Entry 40 days ago, position losing 10%
        entry_date = (datetime.now(UTC).date() - timedelta(days=40)).isoformat()
        conn = _make_db({"AAPL": entry_date})
        # Position: entry=100, current=90 → -10%
        pos = _make_position("AAPL", entry_price=100.0, market_value=90.0, qty=1.0)
        result = analyze_aging(_FakeBroker([pos]), conn=conn)
        assert result is not None
        aged_pos = result.positions[0]
        assert aged_pos.overstay_flag is True
        assert result.overstayed_count == 1

    def test_no_overstay_for_profitable_position(self):
        entry_date = (datetime.now(UTC).date() - timedelta(days=40)).isoformat()
        conn = _make_db({"AAPL": entry_date})
        pos = _make_position("AAPL", entry_price=100.0, market_value=120.0, qty=1.0)
        result = analyze_aging(_FakeBroker([pos]), conn=conn)
        assert result is not None
        assert result.positions[0].overstay_flag is False

    def test_oldest_symbol_populated(self):
        entry_aapl = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
        entry_msft = (datetime.now(UTC).date() - timedelta(days=5)).isoformat()
        conn = _make_db({"AAPL": entry_aapl, "MSFT": entry_msft})
        broker = _FakeBroker([_make_position("AAPL"), _make_position("MSFT")])
        result = analyze_aging(broker, conn=conn)
        assert result is not None
        assert result.oldest_symbol == "AAPL"

    def test_avg_hold_days_none_without_db(self):
        result = analyze_aging(_FakeBroker())
        assert result is not None
        assert result.avg_hold_days is None
