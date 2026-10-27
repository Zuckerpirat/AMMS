"""Standalone simulation loop — no Alpaca required.

Pulls live prices from Yahoo Finance, keeps a $100k virtual portfolio in
a local JSON file, decides buy/sell via the Decision Engine, and expands
the watchlist from WSB mentions via ApeWisdom (no Reddit key needed).

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
from amms.data.wsb_discovery import (
    DiscoveryState,
    WSBDiscoveryConfig,
    format_delta_message,
    maybe_refresh as maybe_refresh_wsb,
)
from amms.data.yahoo import YahooDataClient
from amms.execution.auto_trader import AutoTrader, AutoTraderConfig
from amms.execution.paper_trader import PaperTrader
from amms.filters.universe import UniverseFilter
from amms.notifier import build_notifier

logger = logging.getLogger(__name__)


def _is_market_open(now: datetime | None = None) -> bool:
    """True during US regular hours Mon–Fri (approx; no holidays)."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes <= 21 * 60


def _merged_symbols(
    settings: Settings,
    config: AppConfig,
    wsb_extras: frozenset[str],
) -> list[str]:
    """Static watchlist + WSB-discovered + user-added (via /add)."""
    static = list(config.watchlist)
    try:
        user_extras = sorted(load_user_extras(settings.db_path))
    except Exception:
        user_extras = []
    seen: set[str] = set()
    out: list[str] = []
    for s in static + list(wsb_extras) + user_extras:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _passes_universe(
    symbol: str, data: YahooDataClient, uf: UniverseFilter
) -> tuple[bool, str | None]:
    """Pre-trade quality gate: price, volume, liquidity."""
    bars = data.get_bars(symbol, limit=max(uf.adv_lookback + 5, 40))
    return uf.passes(bars)


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
    universe = config.universe or UniverseFilter()

    # Permissive AutoTrader config so signals actually fire in simulation.
    at_cfg = AutoTraderConfig(
        max_position_pct=0.05,
        max_positions=10,
        cooldown_minutes=30,
        min_confidence=0.50,
        min_score=20.0,
        allow_strong_only=False,
    )
    auto = AutoTrader(trader, data, at_cfg)

    # WSB auto-discovery (ApeWisdom fallback — no Reddit key needed).
    wsb_cfg = config.wsb_discovery or WSBDiscoveryConfig()
    wsb_state = DiscoveryState()

    snap = trader.snapshot()
    notifier.send(
        f"🧪 amms simulate started — ${snap.portfolio_value:,.0f} virtual, "
        f"tick {tick_seconds}s"
        + (f"\n🔍 WSB discovery on (top {wsb_cfg.top_n}, every {wsb_cfg.refresh_hours}h)"
           if wsb_cfg.enabled else "\n🔍 WSB discovery off — enable in config.yaml")
    )
    logger.info("simulate loop running (tick %ds)", tick_seconds)

    tick_n = 0
    try:
        while not stop.is_set():
            # WSB discovery runs even when market is closed, so new names
            # are ready at open. Scanner's own refresh_hours gate protects
            # against hammering ApeWisdom.
            try:
                delta = maybe_refresh_wsb(
                    wsb_state, wsb_cfg,
                    static_watchlist=set(config.watchlist),
                    now_seconds=time.time(),
                )
                if delta.refreshed and (delta.added or delta.removed):
                    notifier.send(format_delta_message(delta))
            except Exception:
                logger.exception("WSB discovery failed")

            if not _is_market_open():
                stop.wait(60)
                continue

            tick_n += 1
            symbols = _merged_symbols(settings, config, wsb_state.extras)

            # Quality gate — only let names through that have enough price
            # and liquidity. Protects against WSB pumping penny-cap garbage.
            allowed: list[str] = []
            dropped: list[str] = []
            for sym in symbols:
                # Already holding? Always re-evaluate so we can SELL.
                if trader.position(sym) is not None:
                    allowed.append(sym)
                    continue
                ok, reason = _passes_universe(sym, data, universe)
                if ok:
                    allowed.append(sym)
                else:
                    dropped.append(f"{sym} ({reason})")

            results = auto.run_watchlist(allowed)
            bought = [r for r in results if r.action == "bought"]
            closed = [r for r in results if r.action == "closed"]

            for r in bought:
                src = "📌" if r.symbol in [s.upper() for s in config.watchlist] else "🔥"
                notifier.send(
                    f"🟢 BOUGHT {src} {r.symbol} {r.qty:.2f}@${r.price:.2f} "
                    f"(score {r.score:+.0f}, conf {r.confidence:.0%})"
                )
            for r in closed:
                notifier.send(
                    f"🔴 CLOSED {r.symbol} {r.qty:.2f}@${r.price:.2f} "
                    f"(score {r.score:+.0f})"
                )

            logger.info(
                "tick %d: %d allowed (%d dropped), %d bought, %d closed",
                tick_n, len(allowed), len(dropped), len(bought), len(closed),
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
