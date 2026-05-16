"""Alpaca Paper Broker — wraps AlpacaClient to match the PaperTrader interface.

This adapter lets the AutoTrader execute against Alpaca's paper-trading
API instead of (or alongside) the local in-memory PaperTrader. Same
buy/sell/close/snapshot signatures, so swapping is transparent.

Safety:
  - Uses the existing `amms.broker.alpaca.AlpacaClient` which refuses any
    URL that is not a paper endpoint (contains "paper-api").
  - This file is paper-only. Live trading requires deliberate changes
    elsewhere (config.PAPER_HOST_MARKER guard removal + explicit opt-in).

Mapping:
  buy(sym, qty, price, reason)         → submit_order(sym, qty, "buy",  market)
  sell(sym, qty, price, reason)        → submit_order(sym, qty, "sell", market)
  close_position(sym, price, reason)   → submit_order(sym, held_qty, "sell")
  snapshot(prices)                     → reads account + positions from Alpaca
  position(sym)                        → reads single position from Alpaca

Limit-order support omitted on purpose — keeping the broker simple for
the initial paper-to-live progression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from amms.broker.alpaca import AlpacaClient, AlpacaError, Order
from amms.execution.paper_trader import PortfolioSnapshot, Position, Trade

logger = logging.getLogger(__name__)


class AlpacaPaperBroker:
    """Broker adapter that routes paper trades through Alpaca's API.

    Compatible with everything the AutoTrader expects of a "trader" object:
    `.buy()`, `.sell()`, `.close_position()`, `.position()`, `.snapshot()`,
    `.save()` (no-op — Alpaca is the source of truth).
    """

    def __init__(self, client: AlpacaClient):
        self.client = client
        self.name = "alpaca-paper"
        self._trade_counter = 0
        self._local_trades: list[Trade] = []   # last-N cache for /ptrades

    # ── Convenience for inbound handlers ──────────────────────────────────

    @classmethod
    def from_settings(cls, settings) -> "AlpacaPaperBroker":
        """Construct from a loaded `Settings` object."""
        client = AlpacaClient(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            base_url=settings.alpaca_base_url,
        )
        return cls(client)

    def close(self) -> None:
        self.client.close()

    # ── Order placement ───────────────────────────────────────────────────

    def buy(self, symbol: str, qty: float, price: float, reason: str = "") -> Trade | None:
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()
        try:
            order = self.client.submit_order(symbol, qty, "buy")
        except AlpacaError as exc:
            logger.warning("Alpaca BUY rejected for %s: %s", symbol, exc)
            return None
        except Exception as exc:
            logger.exception("Alpaca BUY error for %s: %s", symbol, exc)
            return None

        trade = self._order_to_trade(order, price, reason)
        self._local_trades.append(trade)
        logger.info("ALPACA-PAPER BUY  %s × %.4f (order %s)", symbol, qty, order.id)
        return trade

    def sell(self, symbol: str, qty: float, price: float, reason: str = "") -> Trade | None:
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()
        held = self.position(symbol)
        if held is None or held.qty < qty - 1e-9:
            logger.warning(
                "Alpaca SELL rejected — insufficient position for %s: need %.4f, have %.4f",
                symbol, qty, held.qty if held else 0
            )
            return None
        try:
            order = self.client.submit_order(symbol, qty, "sell")
        except AlpacaError as exc:
            logger.warning("Alpaca SELL rejected for %s: %s", symbol, exc)
            return None
        except Exception as exc:
            logger.exception("Alpaca SELL error for %s: %s", symbol, exc)
            return None

        trade = self._order_to_trade(order, price, reason)
        self._local_trades.append(trade)
        logger.info("ALPACA-PAPER SELL %s × %.4f (order %s)", symbol, qty, order.id)
        return trade

    def close_position(self, symbol: str, price: float, reason: str = "") -> Trade | None:
        held = self.position(symbol)
        if held is None or held.qty <= 0:
            return None
        return self.sell(symbol, held.qty, price, reason=reason)

    # ── Read account state from Alpaca ────────────────────────────────────

    def position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        try:
            positions = self.client.get_positions()
        except Exception as exc:
            logger.warning("Could not fetch Alpaca positions: %s", exc)
            return None
        for p in positions:
            if p.symbol.upper() == symbol:
                return Position(
                    symbol=p.symbol.upper(),
                    qty=p.qty,
                    avg_cost=p.avg_entry_price,
                    realized_pnl=0.0,  # Alpaca tracks it differently
                )
        return None

    def snapshot(self, prices: dict[str, float] | None = None) -> PortfolioSnapshot:
        """Build a snapshot from live Alpaca account state."""
        try:
            account = self.client.get_account()
            positions = self.client.get_positions()
        except Exception as exc:
            logger.error("Could not fetch Alpaca state: %s", exc)
            # Return empty snapshot rather than raising
            return PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cash=0.0,
                positions={},
                total_market_value=0.0,
                portfolio_value=0.0,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_return_pct=0.0,
                trade_count=0,
                starting_cash=0.0,
            )

        pos_data = {}
        total_market = 0.0
        total_unrealized = 0.0
        for p in positions:
            sym = p.symbol.upper()
            cur_price = prices.get(sym) if prices else None
            mv = p.market_value
            upnl = p.unrealized_pl
            total_market += mv
            total_unrealized += upnl
            pos_data[sym] = {
                "qty": round(p.qty, 6),
                "avg_cost": round(p.avg_entry_price, 4),
                "market_value": round(mv, 2),
                "unrealized_pnl": round(upnl, 2),
                "pnl_pct": (
                    round((p.avg_entry_price and (mv / (p.avg_entry_price * p.qty) - 1.0) * 100.0) or 0.0, 2)
                    if p.qty > 0 and p.avg_entry_price > 0 else 0.0
                ),
            }

        starting_cash = float(account.raw.get("last_equity", account.equity))
        ret_pct = (account.equity - starting_cash) / starting_cash * 100.0 if starting_cash > 0 else 0.0

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cash=round(account.cash, 2),
            positions=pos_data,
            total_market_value=round(total_market, 2),
            portfolio_value=round(account.equity, 2),
            total_realized_pnl=0.0,  # not directly available
            total_unrealized_pnl=round(total_unrealized, 2),
            total_return_pct=round(ret_pct, 2),
            trade_count=len(self._local_trades),
            starting_cash=round(starting_cash, 2),
        )

    def recent_trades(self, n: int = 10) -> list[Trade]:
        """Local session-only trade log (not Alpaca's full history)."""
        return list(self._local_trades[-n:])

    def save(self, *_args, **_kwargs) -> None:
        """No-op — Alpaca holds the authoritative state."""
        return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _order_to_trade(self, order: Order, price: float, reason: str) -> Trade:
        self._trade_counter += 1
        return Trade(
            id=self._trade_counter,
            timestamp=order.submitted_at,
            symbol=order.symbol.upper(),
            side=order.side,
            qty=order.qty,
            price=order.filled_avg_price if order.filled_avg_price else price,
            total=round(order.qty * (order.filled_avg_price or price), 4),
            commission=0.0,
            reason=f"alpaca:{order.id[:8]} {reason}".strip(),
            portfolio_value_after=0.0,  # Snapshot computed on demand
        )
