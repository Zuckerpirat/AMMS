from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date as Date

from amms.data.bars import Bar
from amms.filters import UniverseFilter
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
    timeframe: str = "1Day"
    universe: UniverseFilter = field(default_factory=UniverseFilter)


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


@dataclass(frozen=True)
class BacktestPosition:
    qty: int
    avg_entry_price: float
    entry_date: str  # ISO date of the first buy that opened this position


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, BacktestPosition] = field(default_factory=dict)

    def equity(self, marks: dict[str, float]) -> float:
        held_value = sum(
            p.qty * marks.get(sym, p.avg_entry_price)
            for sym, p in self.positions.items()
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
    timeframe: str,
) -> dict[str, dict[str, Bar]]:
    """Return {symbol: {date_iso: Bar}} for bars of the given timeframe."""
    out: dict[str, dict[str, Bar]] = {sym: {} for sym in symbols}
    end_iso = end.isoformat() + "T23:59:59Z"
    for sym in symbols:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol = ? AND timeframe = ? AND ts BETWEEN ? AND ? "
            "ORDER BY ts ASC",
            (sym, timeframe, start.isoformat(), end_iso),
        ).fetchall()
        for row in rows:
            d = row["ts"][:10]
            out[sym][d] = Bar(
                symbol=sym,
                timeframe=timeframe,
                ts=row["ts"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
    return out


def _days_between(start_iso: str, end_iso: str) -> int:
    return (Date.fromisoformat(end_iso) - Date.fromisoformat(start_iso)).days


def run_backtest(config: BacktestConfig, conn: sqlite3.Connection) -> BacktestResult:
    """Event-driven backtest. Signals at close; fills at next bar's open.

    Reads bars from the SQLite ``bars`` table at ``config.timeframe``. Skips
    dates with no bars for a given symbol but carries pending orders forward
    to the next day on which a bar exists for that symbol. Enforces
    ``risk.min_hold_days`` symmetrically with the live executor.

    NOTE: the engine groups bars by calendar date and steps day by day, so
    intraday timeframes (5Min, 15Min, etc.) collapse to one bar per day.
    Use this for daily backtests today; full intraday simulation is a
    future enhancement.
    """
    bars = _load_bars(conn, config.symbols, config.start, config.end, config.timeframe)
    all_dates = sorted({d for sym in config.symbols for d in bars[sym]})
    if not all_dates:
        raise ValueError(
            f"No bars in DB for symbols {list(config.symbols)} between "
            f"{config.start} and {config.end} at timeframe {config.timeframe!r}. "
            "Run `amms fetch-bars` first."
        )

    portfolio = Portfolio(cash=config.initial_equity)
    trades: list[Trade] = []
    equity_curve: list[tuple[str, float]] = []
    pending: list[PendingOrder] = []
    bars_running: dict[str, list[Bar]] = {sym: [] for sym in config.symbols}
    last_close: dict[str, float] = {}

    for today in all_dates:
        # 1. Fill yesterday's pending orders at today's open.
        remaining: list[PendingOrder] = []
        for order in pending:
            bar_today = bars[order.symbol].get(today)
            if bar_today is None:
                remaining.append(order)
                continue
            price = bar_today.open
            if order.side == "buy":
                _fill_buy(portfolio, trades, order, price, today)
            else:
                _fill_sell(portfolio, trades, order, price, today)
        pending = remaining

        # 2. Roll running bars forward.
        for sym in config.symbols:
            bar = bars[sym].get(today)
            if bar is not None:
                bars_running[sym].append(bar)
                last_close[sym] = bar.close

        # 3. Evaluate the strategy and queue new orders.
        equity_now = portfolio.equity(last_close)
        for sym in config.symbols:
            symbol_bars = bars_running[sym]
            if len(symbol_bars) < config.strategy.lookback:
                continue
            signal = config.strategy.evaluate(sym, symbol_bars)
            held = portfolio.holds(sym)

            if signal.kind == "buy" and not held:
                passes, _ = config.universe.passes(symbol_bars)
                if not passes:
                    continue  # Universe gate; symbol does not qualify today.
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
                position = portfolio.positions[sym]
                if config.risk.min_hold_days > 0:
                    days_held = _days_between(position.entry_date, today)
                    if days_held < config.risk.min_hold_days:
                        continue  # Hold-time gate; strategy can re-emit tomorrow.
                pending.append(
                    PendingOrder(sym, position.qty, "sell", signal.reason)
                )

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
    existing = portfolio.positions.get(order.symbol)
    if existing is None:
        new_position = BacktestPosition(
            qty=order.qty,
            avg_entry_price=price,
            entry_date=date_iso,
        )
    else:
        new_qty = existing.qty + order.qty
        new_avg = (existing.qty * existing.avg_entry_price + order.qty * price) / new_qty
        # Adding to a position preserves the original entry_date so min_hold_days
        # measures from when the trader first got in, not the latest top-up.
        new_position = BacktestPosition(
            qty=new_qty,
            avg_entry_price=new_avg,
            entry_date=existing.entry_date,
        )
    portfolio.positions[order.symbol] = new_position
    trades.append(Trade(date_iso, order.symbol, "buy", order.qty, price, order.reason))


def _fill_sell(
    portfolio: Portfolio,
    trades: list[Trade],
    order: PendingOrder,
    price: float,
    date_iso: str,
) -> None:
    existing = portfolio.positions.get(order.symbol)
    if existing is None:
        return
    qty = min(order.qty, existing.qty)
    if qty <= 0:
        return
    portfolio.cash += qty * price
    new_qty = existing.qty - qty
    if new_qty == 0:
        del portfolio.positions[order.symbol]
    else:
        portfolio.positions[order.symbol] = BacktestPosition(
            qty=new_qty,
            avg_entry_price=existing.avg_entry_price,
            entry_date=existing.entry_date,
        )
    trades.append(Trade(date_iso, order.symbol, "sell", qty, price, order.reason))
