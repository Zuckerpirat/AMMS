"""Standalone simulation loop — no Alpaca required.

This is the "just run it" mode: a $100k virtual portfolio, Yahoo Finance
for live prices, the Decision Engine for buy/sell decisions, and local
state in a JSON file. Starts, pulls prices for the watchlist, decides,
simulates trades, repeats. Sends Telegram alerts on each trade if
TELEGRAM_BOT_TOKEN/CHAT_ID are set.

Entry point: ``amms simulate`` (see cli.py). Runs until SIGTERM/SIGINT.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from amms.config import AppConfig, Settings
from amms.data.dynamic_watchlist import load as load_user_extras
from amms.data.yahoo import YahooDataClient
from amms.execution.auto_trader import AutoTrader, AutoTraderConfig
from amms.execution.paper_trader import PaperTrader
from amms.notifier import build_notifier

logger = logging.getLogger(__name__)


def _is_market_open(now: datetime | None = None) -> bool:
    """True during US regular hours Mon–Fri (approx; no holidays).

    Market hours: 09:30–16:00 ET → 13:30–20:00 UTC winter, 14:30–21:00 UTC summer.
    Broad 13:30–21:00 UTC window on weekdays covers both.
    """
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes <= 21 * 60


def _merged_symbols(settings: Settings, config: AppConfig) -> list[str]:
    static = list(config.watchlist)
    try:
        extras = sorted(load_user_extras(settings.db_path))
    except Exception:
        extras = []
    seen: set[str] = set()
    out: list[str] = []
    for s in static + extras:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def run_simulation(
    settings: Settings,
    config: AppConfig,
    *,
    tick_seconds: int = 300,
    stop: threading.Event | None = None,
) -> None:
    """Run the standalone simulator until ``stop`` is set."""
    stop = stop or threading.Event()

    def _handler(signum, _frame):
        logger.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    notifier = build_notifier()
    trader = PaperTrader.load()
    data = YahooDataClient(timeframe=config.strategy.timeframe)

    # Permissive config so signals actually fire in simulation. The point
    # here is visible activity, not alpha generation.
    at_cfg = AutoTraderConfig(
        max_position_pct=0.05,
        max_positions=10,
        cooldown_minutes=30,
        min_confidence=0.50,
        min_score=20.0,
        allow_strong_only=False,
    )
    auto = AutoTrader(trader, data, at_cfg)

    snap = trader.snapshot()
    notifier.send(
        f"🧪 amms simulate started — ${snap.portfolio_value:,.0f} virtual, "
        f"{len(_merged_symbols(settings, config))} symbols, tick {tick_seconds}s"
    )
    logger.info("simulate loop running (tick %ds)", tick_seconds)

    tick_n = 0
    try:
        while not stop.is_set():
            if not _is_market_open():
                stop.wait(60)
                continue

            tick_n += 1
            symbols = _merged_symbols(settings, config)
            results = auto.run_watchlist(symbols)

            bought = [r for r in results if r.action == "bought"]
            closed = [r for r in results if r.action == "closed"]

            for r in bought:
                notifier.send(
                    f"🟢 BOUGHT {r.symbol} {r.qty:.2f}@${r.price:.2f} "
                    f"(score {r.score:+.0f}, conf {r.confidence:.0%})"
                )
            for r in closed:
                notifier.send(
                    f"🔴 CLOSED {r.symbol} {r.qty:.2f}@${r.price:.2f} "
                    f"(score {r.score:+.0f})"
                )

            logger.info(
                "tick %d: %d symbols → %d bought, %d closed",
                tick_n, len(results), len(bought), len(closed),
            )
            trader.save()
            stop.wait(tick_seconds)
    finally:
        trader.save()
        data.close()
        snap = trader.snapshot()
        notifier.send(
            f"🧪 amms simulate stopped — portfolio ${snap.portfolio_value:,.2f}, "
            f"realized P&L ${snap.total_realized_pnl:+,.2f}"
        )
