from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date as Date

from amms.risk import RiskConfig, check_buy
from amms.strategy import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    start: Date
    end: Date
    symbols: tuple[str, ...]
    initial_equity: float
    risk: RiskConfig
    strategy: Strategy


@dataclass(frozen=True)
class Trade:
    date: str
    symbol: str
    side: str
    qty: int
    price: float
    reason: str


@dataclass
class PendingOrder:
    symbol: str
    qty: int
    side: str
    reason: str


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, tuple[int, float]] = field(default_factory=dict)

    def equity(self, marks: dict[str, float]) -> float:
        held_value = sum(
            qty * marks.get(sym, entry)
            for sym, (qty, entry) in self.positions.items()
        )
        return self.cash + held_value

    def holds(self, symbol: str) -> bool:
        return symbol in self.positions


@dataclass
class BacktestResult:
    config: BacktestConfig
    portfolio: Portfolio
    trades: list[Trade]
    equity_curve: list[tuple[str, float]]


def _load_bars(
    conn: sqlite3.Connection,
    symbols: tuple[str, ...],
    start: Date,
    end: Date,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {symbol: {date_iso: (open, close)}} for daily bars in range."""
    out: dict[str, dict[str, tuple[float, float]]] = {sym: {} for sym in symbols}
    end_iso = end.isoformat() + "T23:59:59Z"
    for sym in symbols:
        rows = conn.execute(
            "SELECT ts, open, close FROM bars "
            "WHERE symbol = ? AND timeframe = '1Day' AND ts BETWEEN ? AND ? "
            "ORDER BY ts ASC",
            (sym, start.isoformat(), end_iso),
        ).fetchall()
        for row in rows:
            d = row["ts"][:10]
            out[sym][d] = (row["open"], row["close"])
    return out


def run_backtest(config: BacktestConfig, conn: sqlite3.Connection) -> BacktestResult:
    """Event-driven backtest. Signals at close; fills at next bar's open.

    Reads daily bars from the SQLite ``bars`` table. Skips dates with no bars
    for a given symbol but carries pending orders forward to the next day on
    which a bar exists for that symbol.
    """
    bars = _load_bars(conn, config.symbols, config.start, config.end)
    all_dates = sorted({d for sym in config.symbols for d in bars[sym]})
    if not all_dates:
        raise ValueError(
            f"No bars in DB for symbols {list(config.symbols)} between "
            f"{config.start} and {config.end}. Run `amms fetch-bars` first."
        )

    portfolio = Portfolio(cash=config.initial_equity)
    trades: list[Trade] = []
    equity_curve: list[tuple[str, float]] = []
    pending: list[PendingOrder] = []
    closes_running: dict[str, list[float]] = {sym: [] for sym in config.symbols}
    last_close: dict[str, float] = {}

    for today in all_dates:
        # 1. Fill yesterday's pending orders at today's open.
        remaining: list[PendingOrder] = []
        for order in pending:
            bar_today = bars[order.symbol].get(today)
            if bar_today is None:
                remaining.append(order)
                continue
            price = bar_today[0]
            if order.side == "buy":
                _fill_buy(portfolio, trades, order, price, today)
            else:
                _fill_sell(portfolio, trades, order, price, today)
        pending = remaining

        # 2. Roll running closes forward.
        for sym in config.symbols:
            bar = bars[sym].get(today)
            if bar is not None:
                closes_running[sym].append(bar[1])
                last_close[sym] = bar[1]

        # 3. Evaluate the strategy and queue new orders.
        equity_now = portfolio.equity(last_close)
        for sym in config.symbols:
            closes = closes_running[sym]
            if len(closes) < config.strategy.lookback:
                continue
            signal = config.strategy.evaluate(sym, closes)
            held = portfolio.holds(sym)

            if signal.kind == "buy" and not held:
                decision = check_buy(
                    equity=equity_now,
                    price=signal.price,
                    cash=portfolio.cash,
                    open_positions=len(portfolio.positions),
                    daily_pnl_pct=0.0,
                    already_holds=False,
                    config=config.risk,
                )
                if decision.allowed:
                    pending.append(
                        PendingOrder(sym, decision.qty, "buy", signal.reason)
                    )
            elif signal.kind == "sell" and held:
                qty_held, _ = portfolio.positions[sym]
                pending.append(PendingOrder(sym, qty_held, "sell", signal.reason))

        # 4. Mark-to-market at today's close.
        equity_curve.append((today, portfolio.equity(last_close)))

    return BacktestResult(
        config=config,
        portfolio=portfolio,
        trades=trades,
        equity_curve=equity_curve,
    )


def _fill_buy(
    portfolio: Portfolio,
    trades: list[Trade],
    order: PendingOrder,
    price: float,
    date_iso: str,
) -> None:
    cost = order.qty * price
    if cost > portfolio.cash:
        # Couldn't afford the fill price after the gap from close to open.
        return
    portfolio.cash -= cost
    old_qty, old_avg = portfolio.positions.get(order.symbol, (0, 0.0))
    new_qty = old_qty + order.qty
    new_avg = (old_qty * old_avg + order.qty * price) / new_qty
    portfolio.positions[order.symbol] = (new_qty, new_avg)
    trades.append(Trade(date_iso, order.symbol, "buy", order.qty, price, order.reason))


def _fill_sell(
    portfolio: Portfolio,
    trades: list[Trade],
    order: PendingOrder,
    price: float,
    date_iso: str,
) -> None:
    held_qty, avg = portfolio.positions.get(order.symbol, (0, 0.0))
    qty = min(order.qty, held_qty)
    if qty <= 0:
        return
    portfolio.cash += qty * price
    new_qty = held_qty - qty
    if new_qty == 0:
        del portfolio.positions[order.symbol]
    else:
        portfolio.positions[order.symbol] = (new_qty, avg)
    trades.append(Trade(date_iso, order.symbol, "sell", qty, price, order.reason))
