"""Tests for amms.execution.alpaca_broker.AlpacaPaperBroker.

Uses a fake AlpacaClient stub so tests don't hit the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from amms.broker.alpaca import Account, AlpacaError, Order, Position as APos
from amms.execution.alpaca_broker import AlpacaPaperBroker
from amms.execution.paper_trader import PortfolioSnapshot, Position, Trade


class _FakeAlpacaClient:
    """In-memory stand-in for AlpacaClient."""

    def __init__(self):
        self.account = Account(
            equity=100_000.0,
            cash=100_000.0,
            buying_power=200_000.0,
            status="ACTIVE",
            daytrade_count=0,
            raw={"last_equity": 100_000.0},
        )
        self.positions: dict[str, APos] = {}
        self.orders: list[Order] = []
        self.next_order_id = 1
        self.fail_next_order = False

    def get_account(self):
        return self.account

    def get_positions(self):
        return list(self.positions.values())

    def submit_order(self, symbol, qty, side, *, order_type="market", time_in_force="day", client_order_id=None):
        if self.fail_next_order:
            self.fail_next_order = False
            raise AlpacaError("simulated failure")
        oid = f"ord-{self.next_order_id:04d}"
        self.next_order_id += 1
        price = 150.0  # fake fill price
        order = Order(
            id=oid,
            client_order_id=client_order_id or f"cli-{oid}",
            symbol=symbol.upper(),
            side=side,
            qty=qty,
            type=order_type,
            status="filled",
            submitted_at="2025-01-01T00:00:00Z",
            filled_at="2025-01-01T00:00:01Z",
            filled_avg_price=price,
            raw={},
        )
        self.orders.append(order)
        # Simulate position update
        sym = symbol.upper()
        if side == "buy":
            if sym in self.positions:
                p = self.positions[sym]
                new_qty = p.qty + qty
                new_cost = (p.avg_entry_price * p.qty + price * qty) / new_qty
                self.positions[sym] = APos(sym, new_qty, new_cost, new_qty * price, 0.0, {})
            else:
                self.positions[sym] = APos(sym, qty, price, qty * price, 0.0, {})
        elif side == "sell":
            p = self.positions.get(sym)
            if p:
                new_qty = p.qty - qty
                if new_qty < 1e-9:
                    del self.positions[sym]
                else:
                    self.positions[sym] = APos(sym, new_qty, p.avg_entry_price, new_qty * price, 0.0, {})
        return order

    def close(self):
        pass


@pytest.fixture
def broker():
    return AlpacaPaperBroker(_FakeAlpacaClient())


class TestBuy:
    def test_buy_returns_trade(self, broker):
        trade = broker.buy("AAPL", 10, 150.0, reason="test")
        assert trade is not None
        assert isinstance(trade, Trade)
        assert trade.side == "buy"
        assert trade.symbol == "AAPL"

    def test_buy_creates_position(self, broker):
        broker.buy("AAPL", 10, 150.0)
        pos = broker.position("AAPL")
        assert pos is not None
        assert pos.qty == 10

    def test_buy_zero_qty_returns_none(self, broker):
        assert broker.buy("AAPL", 0, 150.0) is None

    def test_buy_failure_returns_none(self, broker):
        broker.client.fail_next_order = True
        assert broker.buy("AAPL", 5, 150.0) is None

    def test_buy_uppercases_symbol(self, broker):
        broker.buy("aapl", 5, 150.0)
        assert broker.position("AAPL") is not None


class TestSell:
    def test_sell_requires_position(self, broker):
        assert broker.sell("AAPL", 5, 150.0) is None

    def test_sell_reduces_position(self, broker):
        broker.buy("AAPL", 10, 150.0)
        broker.sell("AAPL", 4, 160.0)
        pos = broker.position("AAPL")
        assert pos is not None
        assert pos.qty == 6

    def test_sell_full_removes_position(self, broker):
        broker.buy("AAPL", 10, 150.0)
        broker.sell("AAPL", 10, 160.0)
        assert broker.position("AAPL") is None

    def test_sell_more_than_held_returns_none(self, broker):
        broker.buy("AAPL", 5, 150.0)
        assert broker.sell("AAPL", 10, 150.0) is None


class TestClose:
    def test_close_no_position_returns_none(self, broker):
        assert broker.close_position("XYZ", 100.0) is None

    def test_close_sells_all(self, broker):
        broker.buy("AAPL", 12, 150.0)
        trade = broker.close_position("AAPL", 160.0)
        assert trade is not None
        assert trade.qty == 12


class TestSnapshot:
    def test_snapshot_empty(self, broker):
        snap = broker.snapshot()
        assert isinstance(snap, PortfolioSnapshot)
        assert snap.positions == {}

    def test_snapshot_with_positions(self, broker):
        broker.buy("AAPL", 10, 150.0)
        broker.buy("TSLA", 5, 150.0)
        snap = broker.snapshot()
        assert "AAPL" in snap.positions
        assert "TSLA" in snap.positions
        assert snap.positions["AAPL"]["qty"] == 10

    def test_snapshot_portfolio_value_from_account(self, broker):
        snap = broker.snapshot()
        assert snap.portfolio_value == 100_000.0


class TestRecentTrades:
    def test_recent_trades_empty(self, broker):
        assert broker.recent_trades() == []

    def test_recent_trades_tracked(self, broker):
        broker.buy("A", 1, 150.0)
        broker.buy("B", 1, 150.0)
        assert len(broker.recent_trades()) == 2


class TestMisc:
    def test_save_is_noop(self, broker):
        # save() must exist (for compat) and not raise
        broker.save()

    def test_close_releases_client(self, broker):
        broker.close()    # no error

    def test_broker_name(self, broker):
        assert broker.name == "alpaca-paper"
