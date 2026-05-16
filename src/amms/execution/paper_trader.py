"""Paper Trading Engine.

Simulates trade execution with virtual cash. Tracks positions, P&L,
trade history, and portfolio value over time.

Thread-safe via an internal lock. State is persisted to a JSON file
so portfolio survives bot restarts.

NOTE — PRE-LIVE TODO:
    All monetary values here are `float`. Acceptable for paper trading
    but for live trading the entire module must migrate to `Decimal` to
    avoid accumulated rounding errors that cause real broker rejects
    ("insufficient funds" by <0.01¢). This migration touches: cash,
    qty, price, commission, all P&L fields, and the JSON state format.

Usage:
    trader = PaperTrader.load()          # load or create fresh
    trader.buy("AAPL", qty=10, price=150.0, reason="Decision Engine BUY")
    trader.sell("AAPL", qty=5,  price=155.0, reason="Decision Engine SELL")
    snap = trader.snapshot()             # current portfolio state
    trader.save()                        # persist to disk
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CASH = 100_000.0   # starting virtual cash
_STATE_FILE   = Path(os.environ.get("AMMS_PAPER_STATE", "paper_portfolio.json"))


@dataclass
class Trade:
    id: int
    timestamp: str      # ISO-8601 UTC
    symbol: str
    side: str           # "buy" / "sell"
    qty: float
    price: float
    total: float        # qty × price (+ commission)
    commission: float
    reason: str
    portfolio_value_after: float


@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float     # average cost basis
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.qty

    def pnl_pct(self, price: float) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return (price - self.avg_cost) / self.avg_cost * 100.0


@dataclass
class PortfolioSnapshot:
    timestamp: str
    cash: float
    positions: dict[str, dict]      # symbol → {qty, avg_cost, market_value, unrealized_pnl, pnl_pct}
    total_market_value: float       # sum of position market values
    portfolio_value: float          # cash + market value
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_return_pct: float         # vs starting cash
    trade_count: int
    starting_cash: float


class PaperTrader:
    """Thread-safe paper trading portfolio."""

    def __init__(self, starting_cash: float = _DEFAULT_CASH, commission: float = 0.0):
        self._lock = threading.Lock()
        self.starting_cash = starting_cash
        self.commission_rate = commission    # fraction per trade value (e.g. 0.001 = 0.1%)
        self.cash: float = starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self._trade_counter = 0
        self._total_realized_pnl: float = 0.0  # includes closed positions

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path = _STATE_FILE) -> None:
        with self._lock:
            state = {
                "starting_cash": self.starting_cash,
                "commission_rate": self.commission_rate,
                "cash": self.cash,
                "trade_counter": self._trade_counter,
                "total_realized_pnl": self._total_realized_pnl,
                "positions": {
                    sym: {
                        "qty": p.qty,
                        "avg_cost": p.avg_cost,
                        "realized_pnl": p.realized_pnl,
                    }
                    for sym, p in self.positions.items()
                },
                "trades": [asdict(t) for t in self.trades],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        logger.info("Paper portfolio saved to %s", path)

    @classmethod
    def load(cls, path: Path = _STATE_FILE) -> "PaperTrader":
        if not path.exists():
            logger.info("No paper portfolio found at %s — starting fresh", path)
            return cls()
        try:
            state = json.loads(path.read_text())
            trader = cls(
                starting_cash=state.get("starting_cash", _DEFAULT_CASH),
                commission=state.get("commission_rate", 0.0),
            )
            trader.cash = state.get("cash", trader.starting_cash)
            trader._trade_counter = state.get("trade_counter", 0)
            trader._total_realized_pnl = state.get("total_realized_pnl", 0.0)
            for sym, pd in state.get("positions", {}).items():
                trader.positions[sym] = Position(
                    symbol=sym,
                    qty=pd["qty"],
                    avg_cost=pd["avg_cost"],
                    realized_pnl=pd.get("realized_pnl", 0.0),
                )
            for td in state.get("trades", []):
                trader.trades.append(Trade(**td))
            logger.info("Paper portfolio loaded from %s (%d trades)", path, len(trader.trades))
            return trader
        except Exception as exc:
            logger.error("Failed to load paper portfolio: %s — starting fresh", exc)
            return cls()

    # ── Core operations ───────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        qty: float,
        price: float,
        reason: str = "",
    ) -> Trade | None:
        """Execute a paper buy order. Returns None if insufficient cash."""
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()
        commission = qty * price * self.commission_rate
        total = qty * price + commission

        with self._lock:
            if total > self.cash:
                logger.warning(
                    "Insufficient cash for %s: need %.2f, have %.2f",
                    symbol, total, self.cash
                )
                return None

            self.cash -= total

            if symbol in self.positions:
                pos = self.positions[symbol]
                new_qty = pos.qty + qty
                new_cost = (pos.avg_cost * pos.qty + price * qty) / new_qty
                self.positions[symbol] = Position(symbol, new_qty, new_cost, pos.realized_pnl)
            else:
                self.positions[symbol] = Position(symbol, qty, price)

            self._trade_counter += 1
            pv = self._portfolio_value_locked({symbol: price})
            trade = Trade(
                id=self._trade_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                side="buy",
                qty=qty,
                price=price,
                total=round(total, 4),
                commission=round(commission, 4),
                reason=reason,
                portfolio_value_after=round(pv, 2),
            )
            self.trades.append(trade)

        logger.info("PAPER BUY  %s × %.4f @ %.4f | cash left %.2f", symbol, qty, price, self.cash)
        return trade

    def sell(
        self,
        symbol: str,
        qty: float,
        price: float,
        reason: str = "",
    ) -> Trade | None:
        """Execute a paper sell order. Returns None if insufficient position."""
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()

        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None or pos.qty < qty - 1e-9:
                logger.warning(
                    "Insufficient position for %s: need %.4f, have %.4f",
                    symbol, qty, pos.qty if pos else 0
                )
                return None

            commission = qty * price * self.commission_rate
            proceeds = qty * price - commission
            # Realized P&L must include commission so cash and P&L reconcile.
            realized = (price - pos.avg_cost) * qty - commission

            self.cash += proceeds
            self._total_realized_pnl += realized
            new_qty = pos.qty - qty
            new_realized = pos.realized_pnl + realized

            if new_qty < 1e-9:
                del self.positions[symbol]
            else:
                self.positions[symbol] = Position(symbol, new_qty, pos.avg_cost, new_realized)

            self._trade_counter += 1
            pv = self._portfolio_value_locked({symbol: price})
            trade = Trade(
                id=self._trade_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                side="sell",
                qty=qty,
                price=price,
                total=round(proceeds, 4),
                commission=round(commission, 4),
                reason=reason,
                portfolio_value_after=round(pv, 2),
            )
            self.trades.append(trade)

        logger.info("PAPER SELL %s × %.4f @ %.4f | realized P&L %.2f", symbol, qty, price, realized)
        return trade

    def close_position(self, symbol: str, price: float, reason: str = "") -> Trade | None:
        """Sell entire position in a symbol."""
        symbol = symbol.upper()
        with self._lock:
            pos = self.positions.get(symbol)
            qty = pos.qty if pos else 0
        if qty <= 0:
            return None
        return self.sell(symbol, qty, price, reason=reason)

    # ── Query ─────────────────────────────────────────────────────────────

    def snapshot(self, prices: dict[str, float] | None = None) -> PortfolioSnapshot:
        """Return current portfolio state. prices: {symbol: current_price}."""
        prices = prices or {}
        with self._lock:
            pos_data = {}
            total_market = 0.0
            total_unrealized = 0.0
            total_realized = 0.0

            for sym, pos in self.positions.items():
                p = prices.get(sym, pos.avg_cost)  # fallback to cost if no price
                mv = pos.market_value(p)
                upnl = pos.unrealized_pnl(p)
                total_market += mv
                total_unrealized += upnl
                pos_data[sym] = {
                    "qty": round(pos.qty, 6),
                    "avg_cost": round(pos.avg_cost, 4),
                    "market_value": round(mv, 2),
                    "unrealized_pnl": round(upnl, 2),
                    "pnl_pct": round(pos.pnl_pct(p), 2),
                }

            pv = self.cash + total_market
            ret_pct = (pv - self.starting_cash) / self.starting_cash * 100.0

            return PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cash=round(self.cash, 2),
                positions=pos_data,
                total_market_value=round(total_market, 2),
                portfolio_value=round(pv, 2),
                total_realized_pnl=round(self._total_realized_pnl, 2),
                total_unrealized_pnl=round(total_unrealized, 2),
                total_return_pct=round(ret_pct, 2),
                trade_count=len(self.trades),
                starting_cash=self.starting_cash,
            )

    def recent_trades(self, n: int = 10) -> list[Trade]:
        with self._lock:
            return list(self.trades[-n:])

    def position(self, symbol: str) -> Position | None:
        with self._lock:
            return self.positions.get(symbol.upper())

    # ── Internal ──────────────────────────────────────────────────────────

    def _portfolio_value_locked(self, prices: dict[str, float]) -> float:
        """Compute portfolio value (call inside lock)."""
        mv = sum(
            p.qty * prices.get(sym, p.avg_cost)
            for sym, p in self.positions.items()
        )
        return self.cash + mv
