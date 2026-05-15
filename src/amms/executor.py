from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from amms.broker import AlpacaClient
from amms.config import AppConfig
from amms.data import MarketDataClient, upsert_bars
from amms.db import (
    bought_today,
    insert_equity_snapshot,
    latest_buy_submitted_at,
    upsert_features,
    upsert_order,
)
from amms.features import standard_features
from amms.metrics import metrics
from amms.risk import STOP_LOSS_REASON_PREFIX, check_buy, check_sector_cap, check_stop_losses
from amms.strategy import Signal, Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TickResult:
    signals: list[Signal] = field(default_factory=list)
    placed_order_ids: list[str] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def buy_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.kind == "buy"]

    @property
    def sell_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.kind == "sell"]


def record_signal(conn: sqlite3.Connection, strategy_name: str, signal: Signal) -> None:
    ts = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO signals(ts, symbol, strategy, signal, reason, score)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts, symbol, strategy) DO UPDATE SET
            signal = excluded.signal,
            reason = excluded.reason,
            score = excluded.score
        """,
        (ts, signal.symbol, strategy_name, signal.kind, signal.reason, signal.score),
    )


def run_tick(
    *,
    broker: AlpacaClient,
    data: MarketDataClient,
    conn: sqlite3.Connection,
    config: AppConfig,
    strategy: Strategy,
    bars_back: int = 90,
    execute: bool = False,
    paused: bool = False,
) -> TickResult:
    """Run one pass of the strategy across the watchlist.

    Pure orchestration: fetch latest bars, evaluate the strategy, write signals,
    enforce risk rules, and (if ``execute``) place paper BUY orders.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=bars_back * 2)  # cushion for weekends/holidays

    account = broker.get_account()
    insert_equity_snapshot(conn, account)
    metrics.inc("amms_ticks_total")
    metrics.observe("amms_equity_dollars", account.equity)
    metrics.observe("amms_cash_dollars", account.cash)
    metrics.observe("amms_daytrade_count", float(account.daytrade_count))
    metrics.observe("amms_last_tick_unix", datetime.now(UTC).timestamp())
    positions = {p.symbol: p for p in broker.get_positions()}
    open_orders = broker.list_orders(status="open")
    pending_buys = {o.symbol for o in open_orders if o.side == "buy"}
    pending_sells = {o.symbol for o in open_orders if o.side == "sell"}

    feature_ts = datetime.now(UTC).isoformat()
    timeframe = config.strategy.timeframe
    result = TickResult()

    # Stop-loss runs first: synthetic SELL signals for any position that has
    # breached its loss cap. They flow through _process_sell_signals like
    # strategy sells but bypass min_hold_days (handled by reason-prefix check).
    for trigger in check_stop_losses(
        positions=list(positions.values()), config=config.risk
    ):
        sl_signal = Signal(
            symbol=trigger.symbol,
            kind="sell",
            reason=trigger.reason,
            price=trigger.current_price,
            score=trigger.loss_pct,
        )
        result.signals.append(sl_signal)
        record_signal(conn, "stop_loss", sl_signal)
        metrics.labeled_inc("amms_stop_loss_triggers_total", {"kind": trigger.kind})
    bars_by_symbol: dict[str, list] = {}
    for symbol in config.watchlist:
        bars = data.get_bars(
            symbol,
            timeframe,
            start.date().isoformat(),
            end.date().isoformat(),
            limit=bars_back,
        )
        upsert_bars(conn, bars)
        bars_by_symbol[symbol] = bars
        features = standard_features(bars)
        upsert_features(conn, feature_ts, symbol, features)
        signal = strategy.evaluate(symbol, bars)
        result.signals.append(signal)
        record_signal(conn, strategy.name, signal)

    # Highest-scoring buys first so max_buys_per_tick keeps the best candidates.
    ranked_buys = sorted(result.buy_signals, key=lambda s: s.score, reverse=True)
    if config.risk.max_buys_per_tick is not None:
        ranked_buys = ranked_buys[: config.risk.max_buys_per_tick]
    eligible_buy_symbols = {s.symbol for s in ranked_buys}

    top_n = config.risk.max_buys_per_tick
    for signal in result.buy_signals:
        if signal.symbol not in eligible_buy_symbols:
            result.blocked.append(
                (signal.symbol, f"score {signal.score:.4f} not in top-{top_n}")
            )

    open_positions_count = len(positions)
    for signal in ranked_buys:
        if paused:
            result.blocked.append((signal.symbol, "paused"))
            continue
        if signal.symbol in pending_buys:
            result.blocked.append((signal.symbol, "open buy order already pending"))
            continue
        passes, reason = config.universe.passes(bars_by_symbol.get(signal.symbol, []))
        if not passes:
            result.blocked.append((signal.symbol, f"universe filter: {reason}"))
            continue
        if config.universe.require_tradable:
            asset = broker.get_asset(signal.symbol)
            passes_a, asset_reason = config.universe.passes_asset(asset)
            if not passes_a:
                result.blocked.append((signal.symbol, f"asset filter: {asset_reason}"))
                continue
        sector_block = check_sector_cap(
            symbol=signal.symbol,
            positions=list(positions.values()),
            total_equity=account.equity,
            config=config.risk,
        )
        if sector_block is not None:
            result.blocked.append((signal.symbol, sector_block.reason))
            continue
        decision = check_buy(
            equity=account.equity,
            price=signal.price,
            cash=account.cash,
            open_positions=open_positions_count,
            daily_pnl_pct=0.0,
            already_holds=signal.symbol in positions,
            config=config.risk,
        )
        if not decision.allowed:
            result.blocked.append((signal.symbol, decision.reason))
            continue
        if not execute:
            result.blocked.append((signal.symbol, f"dry-run: would buy {decision.qty}"))
            continue
        order = broker.submit_order(signal.symbol, decision.qty, "buy")
        upsert_order(conn, order)
        result.placed_order_ids.append(order.id)
        pending_buys.add(signal.symbol)
        open_positions_count += 1
        metrics.labeled_inc("amms_orders_total", {"side": "buy"})

    _process_sell_signals(
        broker=broker,
        conn=conn,
        config=config,
        result=result,
        positions=positions,
        pending_sells=pending_sells,
        account=account,
        execute=execute,
    )
    return result


def _process_sell_signals(
    *,
    broker: AlpacaClient,
    conn: sqlite3.Connection,
    config: AppConfig,
    result: TickResult,
    positions: dict,
    pending_sells: set[str],
    account,
    execute: bool,
) -> None:
    pdt_locked = (
        account.equity < config.risk.pdt_min_equity
        and account.daytrade_count >= config.risk.pdt_max_day_trades
    )
    for signal in result.sell_signals:
        if signal.symbol not in positions:
            result.blocked.append(
                (signal.symbol, "sell signal but no position held")
            )
            continue
        if signal.symbol in pending_sells:
            result.blocked.append(
                (signal.symbol, "open sell order already pending")
            )
            continue
        is_stop_loss = signal.reason.startswith(STOP_LOSS_REASON_PREFIX)
        if config.risk.min_hold_days > 0 and not is_stop_loss:
            days_held = _days_since_last_buy(conn, signal.symbol)
            if days_held is not None and days_held < config.risk.min_hold_days:
                result.blocked.append(
                    (
                        signal.symbol,
                        f"held {days_held}d < min_hold_days {config.risk.min_hold_days}d",
                    )
                )
                continue
        if pdt_locked and _bought_today(conn, signal.symbol):
            result.blocked.append(
                (
                    signal.symbol,
                    f"PDT lock: equity ${account.equity:,.0f} < ${config.risk.pdt_min_equity:,.0f} "
                    f"and daytrade_count={account.daytrade_count}",
                )
            )
            continue
        qty = positions[signal.symbol].qty
        if qty <= 0:
            result.blocked.append((signal.symbol, "position qty is zero"))
            continue
        if not execute:
            result.blocked.append(
                (signal.symbol, f"dry-run: would sell {qty:g}")
            )
            continue
        order = broker.submit_order(signal.symbol, qty, "sell")
        upsert_order(conn, order)
        result.placed_order_ids.append(order.id)
        pending_sells.add(signal.symbol)
        metrics.labeled_inc("amms_orders_total", {"side": "sell"})


def _bought_today(conn: sqlite3.Connection, symbol: str) -> bool:
    return bought_today(conn, symbol)


def _days_since_last_buy(conn: sqlite3.Connection, symbol: str) -> int | None:
    ts = latest_buy_submitted_at(conn, symbol)
    if ts is None:
        return None
    last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (datetime.now(UTC) - last).days


def build_daily_summary(broker: AlpacaClient, conn: sqlite3.Connection) -> str:
    """Compose a short Telegram-friendly end-of-day summary."""
    account = broker.get_account()
    positions = broker.get_positions()
    today_iso = datetime.now(UTC).date().isoformat()
    orders_today = conn.execute(
        "SELECT count(*) FROM orders WHERE substr(submitted_at, 1, 10) = ?",
        (today_iso,),
    ).fetchone()[0]
    lines = [
        f"amms daily summary {today_iso}",
        f"Equity: ${account.equity:,.2f}",
        f"Cash:   ${account.cash:,.2f}",
        f"Open positions: {len(positions)}",
        f"Orders today:   {orders_today}",
    ]
    return "\n".join(lines)
