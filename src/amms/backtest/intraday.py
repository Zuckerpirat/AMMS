"""Timestamp-stepping backtester for intraday timeframes.

The default engine in ``amms.backtest.engine`` groups bars by calendar
date. That's fine for daily strategies but loses intraday granularity. This
module steps by raw bar timestamp instead, so 5Min / 15Min / 1Hour bars
produce one tick each. Signals form at each bar's close; fills happen at
the *next* bar's open for that symbol.

Risk / universe / min_hold_days behave identically. min_hold_days is
counted in *days* between bar timestamps (so the same threshold means the
same thing across timeframes).
"""

from __future__ import annotations

import sqlite3
from datetime import date as Date
from datetime import datetime

from amms.backtest.engine import (
    BacktestConfig,
    BacktestPosition,
    BacktestResult,
    PendingOrder,
    Portfolio,
    Trade,
    _fill_buy,
    _fill_sell,
)
from amms.data.bars import Bar
from amms.risk import check_buy


def _load_bars_ordered(
    conn: sqlite3.Connection,
    symbols: tuple[str, ...],
    start: Date,
    end: Date,
    timeframe: str,
) -> dict[str, list[Bar]]:
    out: dict[str, list[Bar]] = {sym: [] for sym in symbols}
    end_iso = end.isoformat() + "T23:59:59Z"
    for sym in symbols:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol = ? AND timeframe = ? AND ts BETWEEN ? AND ? "
            "ORDER BY ts ASC",
            (sym, timeframe, start.isoformat(), end_iso),
        ).fetchall()
        out[sym] = [
            Bar(
                symbol=sym,
                timeframe=timeframe,
                ts=row["ts"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for row in rows
        ]
    return out


def _ts_to_date(ts: str) -> Date:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()


def run_intraday_backtest(
    config: BacktestConfig, conn: sqlite3.Connection
) -> BacktestResult:
    """Timestamp-stepping intraday backtest. Caveat: ``equity_curve`` is
    sampled once per *bar timestamp* across all symbols, not once per
    calendar day, which can produce a very long curve on small timeframes.
    """
    bars_by_sym = _load_bars_ordered(
        conn, config.symbols, config.start, config.end, config.timeframe
    )
    timestamps = sorted({b.ts for sym in config.symbols for b in bars_by_sym[sym]})
    if not timestamps:
        raise ValueError(
            f"No bars in DB for symbols {list(config.symbols)} between "
            f"{config.start} and {config.end} at timeframe {config.timeframe!r}."
        )

    # Per-symbol index pointer into bars_by_sym + reverse lookup of bar by ts.
    bar_at: dict[tuple[str, str], Bar] = {
        (sym, b.ts): b for sym in config.symbols for b in bars_by_sym[sym]
    }
    next_bar_ts: dict[tuple[str, str], str] = {}
    for sym in config.symbols:
        seq = bars_by_sym[sym]
        for i in range(len(seq) - 1):
            next_bar_ts[(sym, seq[i].ts)] = seq[i + 1].ts

    portfolio = Portfolio(cash=config.initial_equity)
    trades: list[Trade] = []
    equity_curve: list[tuple[str, float]] = []
    pending: list[tuple[str, PendingOrder]] = []  # (target_ts, order)
    last_close: dict[str, float] = {}
    bars_running: dict[str, list[Bar]] = {sym: [] for sym in config.symbols}

    for now_ts in timestamps:
        # 1. Fill any pending order targeting this ts.
        remaining: list[tuple[str, PendingOrder]] = []
        for target_ts, order in pending:
            bar = bar_at.get((order.symbol, target_ts))
            if target_ts != now_ts or bar is None:
                # Either still in the future or the symbol has no bar there;
                # keep waiting.
                if target_ts > now_ts:
                    remaining.append((target_ts, order))
                continue
            price = bar.open
            if order.side == "buy":
                _fill_buy(portfolio, trades, order, price, target_ts[:10])
            else:
                _fill_sell(portfolio, trades, order, price, target_ts[:10])
        pending = remaining

        # 2. Process bars present at now_ts.
        for sym in config.symbols:
            bar = bar_at.get((sym, now_ts))
            if bar is None:
                continue
            bars_running[sym].append(bar)
            last_close[sym] = bar.close

            if len(bars_running[sym]) < config.strategy.lookback:
                continue
            signal = config.strategy.evaluate(sym, bars_running[sym])
            held = portfolio.holds(sym)
            equity_now = portfolio.equity(last_close)

            if signal.kind == "buy" and not held:
                passes, _ = config.universe.passes(bars_running[sym])
                if not passes:
                    continue
                decision = check_buy(
                    equity=equity_now,
                    price=signal.price,
                    cash=portfolio.cash,
                    open_positions=len(portfolio.positions),
                    daily_pnl_pct=0.0,
                    already_holds=False,
                    config=config.risk,
                )
                if not decision.allowed:
                    continue
                target = next_bar_ts.get((sym, now_ts))
                if target is None:
                    continue
                pending.append(
                    (target, PendingOrder(sym, decision.qty, "buy", signal.reason))
                )
            elif signal.kind == "sell" and held:
                position: BacktestPosition = portfolio.positions[sym]
                if config.risk.min_hold_days > 0:
                    today = _ts_to_date(now_ts)
                    entry = _ts_to_date(position.entry_date + "T00:00:00Z")
                    if (today - entry).days < config.risk.min_hold_days:
                        continue
                target = next_bar_ts.get((sym, now_ts))
                if target is None:
                    continue
                pending.append(
                    (target, PendingOrder(sym, position.qty, "sell", signal.reason))
                )

        equity_curve.append((now_ts, portfolio.equity(last_close)))

    return BacktestResult(
        config=config,
        portfolio=portfolio,
        trades=trades,
        equity_curve=equity_curve,
    )
