"""Tests for amms.execution.paper_trader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from amms.execution.paper_trader import PaperTrader, Position, PortfolioSnapshot, Trade


class TestInitialization:
    def test_default_cash(self):
        t = PaperTrader()
        assert t.cash == 100_000.0

    def test_custom_cash(self):
        t = PaperTrader(starting_cash=50_000.0)
        assert t.cash == 50_000.0

    def test_no_positions_initially(self):
        t = PaperTrader()
        assert len(t.positions) == 0

    def test_no_trades_initially(self):
        t = PaperTrader()
        assert len(t.trades) == 0


class TestBuy:
    def test_buy_reduces_cash(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 150.0)
        assert t.cash == 100_000.0 - 1500.0

    def test_buy_creates_position(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 150.0)
        assert "AAPL" in t.positions
        assert t.positions["AAPL"].qty == 10
        assert t.positions["AAPL"].avg_cost == 150.0

    def test_buy_returns_trade(self):
        t = PaperTrader()
        trade = t.buy("AAPL", 10, 150.0)
        assert trade is not None
        assert isinstance(trade, Trade)
        assert trade.side == "buy"
        assert trade.symbol == "AAPL"

    def test_buy_insufficient_cash_returns_none(self):
        t = PaperTrader(starting_cash=100.0)
        result = t.buy("AAPL", 10, 150.0)
        assert result is None
        assert t.cash == 100.0  # unchanged

    def test_buy_averages_cost(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        t.buy("AAPL", 10, 120.0)
        pos = t.positions["AAPL"]
        assert pos.qty == 20
        assert pos.avg_cost == 110.0

    def test_buy_symbol_uppercased(self):
        t = PaperTrader()
        t.buy("aapl", 5, 100.0)
        assert "AAPL" in t.positions

    def test_buy_zero_qty_returns_none(self):
        t = PaperTrader()
        assert t.buy("AAPL", 0, 100.0) is None

    def test_buy_records_trade(self):
        t = PaperTrader()
        t.buy("AAPL", 5, 100.0)
        assert len(t.trades) == 1

    def test_buy_with_commission(self):
        t = PaperTrader(commission=0.001)
        t.buy("AAPL", 10, 100.0)
        expected_cash = 100_000.0 - (10 * 100.0 * 1.001)
        assert abs(t.cash - expected_cash) < 0.01


class TestSell:
    def test_sell_increases_cash(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        t.sell("AAPL", 5, 110.0)
        expected_cash = 100_000.0 - 1000.0 + 550.0
        assert abs(t.cash - expected_cash) < 0.01

    def test_sell_reduces_position(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        t.sell("AAPL", 4, 110.0)
        assert t.positions["AAPL"].qty == 6

    def test_sell_full_position_removes_it(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        t.sell("AAPL", 10, 110.0)
        assert "AAPL" not in t.positions

    def test_sell_no_position_returns_none(self):
        t = PaperTrader()
        result = t.sell("AAPL", 5, 100.0)
        assert result is None

    def test_sell_more_than_held_returns_none(self):
        t = PaperTrader()
        t.buy("AAPL", 5, 100.0)
        result = t.sell("AAPL", 10, 100.0)
        assert result is None

    def test_sell_returns_trade(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        trade = t.sell("AAPL", 5, 110.0)
        assert trade is not None
        assert trade.side == "sell"

    def test_sell_realized_pnl_tracked(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        t.sell("AAPL", 10, 110.0)
        # Position removed; realized P&L = (110-100)*10 = 100
        snap = t.snapshot()
        assert snap.total_realized_pnl == 100.0


class TestClosePosition:
    def test_close_sells_all(self):
        t = PaperTrader()
        t.buy("MSFT", 20, 300.0)
        trade = t.close_position("MSFT", 310.0)
        assert trade is not None
        assert trade.qty == 20
        assert "MSFT" not in t.positions

    def test_close_no_position_returns_none(self):
        t = PaperTrader()
        assert t.close_position("MSFT", 300.0) is None


class TestSnapshot:
    def test_snapshot_returns_snapshot(self):
        t = PaperTrader()
        snap = t.snapshot()
        assert isinstance(snap, PortfolioSnapshot)

    def test_snapshot_initial_values(self):
        t = PaperTrader(starting_cash=50_000.0)
        snap = t.snapshot()
        assert snap.cash == 50_000.0
        assert snap.portfolio_value == 50_000.0
        assert snap.total_return_pct == 0.0
        assert snap.trade_count == 0

    def test_snapshot_with_prices(self):
        t = PaperTrader()
        t.buy("AAPL", 10, 100.0)
        snap = t.snapshot({"AAPL": 120.0})
        assert snap.total_market_value == 1200.0
        assert snap.total_unrealized_pnl == 200.0
        assert snap.positions["AAPL"]["pnl_pct"] == 20.0

    def test_snapshot_return_pct(self):
        t = PaperTrader(starting_cash=10_000.0)
        t.buy("X", 10, 100.0)
        t.sell("X", 10, 110.0)
        snap = t.snapshot()
        assert abs(snap.total_return_pct - 1.0) < 0.01  # 1% gain

    def test_snapshot_no_positions(self):
        t = PaperTrader()
        snap = t.snapshot()
        assert snap.positions == {}
        assert snap.total_market_value == 0.0


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test_portfolio.json"
            t = PaperTrader(starting_cash=25_000.0)
            t.buy("GOOG", 2, 175.0)
            t.save(path)

            t2 = PaperTrader.load(path)
            assert abs(t2.cash - t.cash) < 0.01
            assert "GOOG" in t2.positions
            assert t2.positions["GOOG"].qty == 2
            assert len(t2.trades) == 1

    def test_load_missing_file_creates_fresh(self):
        t = PaperTrader.load(Path("/tmp/amms_nonexistent_99999.json"))
        assert t.cash == 100_000.0

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "portfolio.json"
            t = PaperTrader()
            t.save(path)
            assert path.exists()


class TestRecentTrades:
    def test_recent_trades_empty(self):
        t = PaperTrader()
        assert t.recent_trades() == []

    def test_recent_trades_limited(self):
        t = PaperTrader()
        for i in range(15):
            t.buy("AAPL", 1, 100.0 + i)
            t.sell("AAPL", 1, 101.0 + i)
        trades = t.recent_trades(5)
        assert len(trades) == 5

    def test_trade_ids_increment(self):
        t = PaperTrader()
        t1 = t.buy("A", 1, 10.0)
        t2 = t.buy("B", 1, 20.0)
        assert t2.id == t1.id + 1


class TestPosition:
    def test_position_none_if_not_held(self):
        t = PaperTrader()
        assert t.position("AAPL") is None

    def test_position_returns_position(self):
        t = PaperTrader()
        t.buy("AAPL", 5, 100.0)
        p = t.position("AAPL")
        assert p is not None
        assert isinstance(p, Position)

    def test_position_market_value(self):
        p = Position("AAPL", 10, 100.0)
        assert p.market_value(120.0) == 1200.0

    def test_position_unrealized_pnl(self):
        p = Position("AAPL", 10, 100.0)
        assert p.unrealized_pnl(110.0) == 100.0

    def test_position_pnl_pct(self):
        p = Position("AAPL", 10, 100.0)
        assert abs(p.pnl_pct(110.0) - 10.0) < 0.01
