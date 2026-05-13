from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from amms.broker import AlpacaClient
from amms.config import AppConfig
from amms.data import MarketDataClient, upsert_bars
from amms.db import insert_equity_snapshot, upsert_features, upsert_order
from amms.features import standard_features
from amms.risk import check_buy
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
) -> TickResult:
    """Run one pass of the strategy across the watchlist.

    Pure orchestration: fetch latest bars, evaluate the strategy, write signals,
    enforce risk rules, and (if ``execute``) place paper BUY orders.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=bars_back * 2)  # cushion for weekends/holidays

    account = broker.get_account()
    insert_equity_snapshot(conn, account)
    positions = {p.symbol: p for p in broker.get_positions()}
    pending_buys = {
        o.symbol for o in broker.list_orders(status="open") if o.side == "buy"
    }

    feature_ts = datetime.now(UTC).isoformat()
    result = TickResult()
    for symbol in config.watchlist:
        bars = data.get_bars(
            symbol,
            "1Day",
            start.date().isoformat(),
            end.date().isoformat(),
            limit=bars_back,
        )
        upsert_bars(conn, bars)
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
        if signal.symbol in pending_buys:
            result.blocked.append((signal.symbol, "open buy order already pending"))
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
    return result


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
