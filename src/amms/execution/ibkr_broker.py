"""IBKR Broker — wraps ib_insync to match the PaperTrader interface.

Connects to Interactive Brokers TWS or IB Gateway via the TWS API.
Drop-in replacement for AlpacaPaperBroker and PaperTrader.

Environment variables:
  IBKR_HOST       IB Gateway host (default: ib-gateway)
  IBKR_PORT       IB Gateway port (default: 4003 for paper, 4001 for live)
  IBKR_CLIENT_ID  TWS client ID (default: 1)
  IBKR_PAPER      "true" for paper trading, "false" for live (default: true)

Requires:
  pip install ib_insync
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from amms.execution.paper_trader import PortfolioSnapshot, Position, Trade

logger = logging.getLogger(__name__)

_PAPER_PORT = 4003
_LIVE_PORT = 4001
_STARTING_CASH = 1_000_000.0  # IBKR paper default


class IBKRBroker:
    """Broker adapter for Interactive Brokers via ib_insync.

    Routes orders through IB Gateway (paper or live). Same interface as
    AlpacaPaperBroker so swapping is transparent to AutoTrader/Scheduler.
    """

    def __init__(self, ib: object, account: str = "", paper: bool = True) -> None:
        self._ib = ib
        self._account = account
        self._paper = paper
        self.name = "ibkr-paper" if paper else "ibkr-live"
        self._trade_counter = 0
        self._local_trades: list[Trade] = []

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def connect(
        cls,
        host: str = "ib-gateway",
        port: int = _PAPER_PORT,
        client_id: int = 1,
        paper: bool = True,
    ) -> "IBKRBroker":
        """Connect to IB Gateway or TWS and return a broker instance."""
        try:
            from ib_insync import IB, util
            util.logToConsole(logging.WARNING)
        except ImportError as exc:
            raise ImportError(
                "ib_insync not installed. Add it with: pip install ib_insync"
            ) from exc

        ib = IB()
        ib.connect(host, port, clientId=client_id, readonly=False, timeout=15)
        accounts = ib.managedAccounts()
        account = accounts[0] if accounts else ""
        logger.info(
            "IBKR connected: host=%s port=%d account=%s paper=%s",
            host, port, account, paper,
        )
        return cls(ib, account=account, paper=paper)

    @classmethod
    def from_env(cls) -> "IBKRBroker":
        """Construct from environment variables."""
        host = os.environ.get("IBKR_HOST", "ib-gateway")
        paper = os.environ.get("IBKR_PAPER", "true").lower() in ("true", "1", "yes")
        default_port = _PAPER_PORT if paper else _LIVE_PORT
        port = int(os.environ.get("IBKR_PORT", str(default_port)))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        return cls.connect(host=host, port=port, client_id=client_id, paper=paper)

    def close(self) -> None:
        try:
            self._ib.disconnect()
        except Exception:
            pass

    # ── Order placement ───────────────────────────────────────────────────

    def buy(self, symbol: str, qty: float, price: float, reason: str = "") -> Trade | None:
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()
        try:
            from ib_insync import MarketOrder, Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = MarketOrder("BUY", qty, account=self._account)
            self._ib.placeOrder(contract, order)
            self._ib.sleep(1.0)
        except Exception as exc:
            logger.warning("IBKR BUY rejected for %s: %s", symbol, exc)
            return None

        t = self._make_trade(symbol, "buy", qty, price, reason)
        self._local_trades.append(t)
        logger.info("IBKR BUY  %s × %.4f @ %.2f", symbol, qty, price)
        return t

    def sell(self, symbol: str, qty: float, price: float, reason: str = "") -> Trade | None:
        if qty <= 0 or price <= 0:
            return None
        symbol = symbol.upper()
        held = self.position(symbol)
        if held is None or held.qty < qty - 1e-9:
            logger.warning(
                "IBKR SELL rejected — insufficient position for %s: need %.4f, have %.4f",
                symbol, qty, held.qty if held else 0,
            )
            return None
        try:
            from ib_insync import MarketOrder, Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order = MarketOrder("SELL", qty, account=self._account)
            self._ib.placeOrder(contract, order)
            self._ib.sleep(1.0)
        except Exception as exc:
            logger.warning("IBKR SELL rejected for %s: %s", symbol, exc)
            return None

        t = self._make_trade(symbol, "sell", qty, price, reason)
        self._local_trades.append(t)
        logger.info("IBKR SELL %s × %.4f @ %.2f", symbol, qty, price)
        return t

    def close_position(self, symbol: str, price: float, reason: str = "") -> Trade | None:
        held = self.position(symbol)
        if held is None or held.qty <= 0:
            return None
        return self.sell(symbol, held.qty, price, reason=reason)

    # ── Read account state ────────────────────────────────────────────────

    def position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        try:
            for pos in self._ib.positions(account=self._account):
                if pos.contract.symbol.upper() == symbol:
                    qty = float(pos.position)
                    if qty == 0:
                        return None
                    avg_cost = float(pos.avgCost)  # cost per share for stocks
                    return Position(
                        symbol=symbol,
                        qty=qty,
                        avg_cost=avg_cost,
                        realized_pnl=0.0,
                    )
        except Exception as exc:
            logger.warning("IBKR position lookup failed for %s: %s", symbol, exc)
        return None

    def snapshot(self, prices: dict[str, float] | None = None) -> PortfolioSnapshot:
        try:
            summary = {
                v.tag: float(v.value)
                for v in self._ib.accountSummary(account=self._account)
                if v.currency in ("USD", "BASE", "")
            }
            positions = [p for p in self._ib.positions(account=self._account) if float(p.position) != 0]
        except Exception as exc:
            logger.error("IBKR snapshot failed: %s", exc)
            return PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cash=0.0, positions={}, total_market_value=0.0,
                portfolio_value=0.0, total_realized_pnl=0.0,
                total_unrealized_pnl=0.0, total_return_pct=0.0,
                trade_count=0, starting_cash=0.0,
            )

        cash = summary.get("TotalCashBalance", summary.get("CashBalance", 0.0))
        net_liq = summary.get("NetLiquidation", cash)
        unrealized = summary.get("UnrealizedPnL", 0.0)
        realized = summary.get("RealizedPnL", 0.0)

        pos_data: dict[str, dict] = {}
        total_market = 0.0
        for pos in positions:
            sym = pos.contract.symbol.upper()
            qty = float(pos.position)
            avg_cost = float(pos.avgCost)
            cur_price = (prices or {}).get(sym, avg_cost)
            mv = qty * cur_price
            upnl = mv - qty * avg_cost
            total_market += mv
            pos_data[sym] = {
                "qty": round(qty, 6),
                "avg_cost": round(avg_cost, 4),
                "market_value": round(mv, 2),
                "unrealized_pnl": round(upnl, 2),
                "pnl_pct": round((cur_price / avg_cost - 1.0) * 100.0, 2) if avg_cost > 0 else 0.0,
            }

        ret_pct = (net_liq - _STARTING_CASH) / _STARTING_CASH * 100.0 if _STARTING_CASH > 0 else 0.0

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cash=round(cash, 2),
            positions=pos_data,
            total_market_value=round(total_market, 2),
            portfolio_value=round(net_liq, 2),
            total_realized_pnl=round(realized, 2),
            total_unrealized_pnl=round(unrealized, 2),
            total_return_pct=round(ret_pct, 2),
            trade_count=len(self._local_trades),
            starting_cash=_STARTING_CASH,
        )

    def get_clock(self) -> None:
        """IBKR has no simple clock API — return None to use local time fallback."""
        return None

    def recent_trades(self, n: int = 10) -> list[Trade]:
        return list(self._local_trades[-n:])

    def save(self, *_args, **_kwargs) -> None:
        return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _make_trade(self, symbol: str, side: str, qty: float, price: float, reason: str) -> Trade:
        self._trade_counter += 1
        return Trade(
            id=self._trade_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            total=round(qty * price, 4),
            commission=0.0,
            reason=reason,
        )
