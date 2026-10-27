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
from amms.data.dynamic_watchlist import (
    add as add_user_extra,
    remove as remove_user_extra,
)
from amms.notifier import build_notifier
from amms.notifier.inbound import TelegramInbound

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

    # Telegram inbound — small handler set that works against PaperTrader.
    inbound = _start_telegram_inbound(trader, data, settings, config, wsb_state)

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
        if inbound is not None:
            inbound.stop()
        trader.save()
        data.close()
        snap = trader.snapshot()
        notifier.send(
            f"🧪 amms simulate stopped — portfolio ${snap.portfolio_value:,.2f}, "
            f"realized P&L ${snap.total_realized_pnl:+,.2f}"
        )


def _start_telegram_inbound(
    trader: PaperTrader,
    data: YahooDataClient,
    settings: Settings,
    config: AppConfig,
    wsb_state: DiscoveryState,
) -> TelegramInbound | None:
    """Wire a small set of simulate-friendly commands to Telegram.

    Only commands that work directly against ``PaperTrader`` + Yahoo are
    exposed here. Everything that needs an Alpaca account (/orders,
    /lastorders, /schedstart …) stays out.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None

    def _prices_for(symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in symbols:
            try:
                bars = data.get_bars(s, limit=2)
                if bars:
                    out[s] = float(bars[-1].close)
            except Exception:
                pass
        return out

    def _portfolio(_args: list[str]) -> str:
        prices = _prices_for(list(trader.positions.keys()))
        snap = trader.snapshot(prices)
        ret_arrow = "▲" if snap.total_return_pct >= 0 else "▼"
        lines = [
            "══ Paper Portfolio ══",
            "",
            f"  Cash:          ${snap.cash:>12,.2f}",
            f"  Market value:  ${snap.total_market_value:>12,.2f}",
            f"  Total value:   ${snap.portfolio_value:>12,.2f}",
            f"  Return:        {ret_arrow} {snap.total_return_pct:+.2f}%  "
            f"(vs ${snap.starting_cash:,.0f} start)",
            f"  Realized P&L:  ${snap.total_realized_pnl:>+12,.2f}",
            f"  Unrealized:    ${snap.total_unrealized_pnl:>+12,.2f}",
            f"  Trades done:   {snap.trade_count}",
        ]
        if snap.positions:
            lines += ["", "  Open positions:"]
            for sym, pd in sorted(snap.positions.items()):
                arrow = "▲" if pd["unrealized_pnl"] >= 0 else "▼"
                lines.append(
                    f"    {sym:<6} {pd['qty']:>8.2f} @ ${pd['avg_cost']:.2f}  "
                    f"MV=${pd['market_value']:,.2f}  "
                    f"P&L {arrow}{pd['unrealized_pnl']:+,.2f} ({pd['pnl_pct']:+.1f}%)"
                )
        else:
            lines += ["", "  No open positions."]
        return "\n".join(lines)

    def _status(_args: list[str]) -> str:
        prices = _prices_for(list(trader.positions.keys()))
        snap = trader.snapshot(prices)
        return (
            f"equity: ${snap.portfolio_value:,.2f} "
            f"({snap.total_return_pct:+.2f}%)\n"
            f"cash: ${snap.cash:,.2f}\n"
            f"open positions: {len(snap.positions)}\n"
            f"trades done: {snap.trade_count}"
        )

    def _positions(_args: list[str]) -> str:
        if not trader.positions:
            return "no open positions"
        prices = _prices_for(list(trader.positions.keys()))
        lines = []
        for sym, pos in sorted(trader.positions.items()):
            px = prices.get(sym, pos.avg_cost)
            pnl = pos.unrealized_pnl(px)
            pct = pos.pnl_pct(px)
            arrow = "▲" if pnl >= 0 else "▼"
            lines.append(
                f"{sym:<6} {pos.qty:>8.2f} @ ${pos.avg_cost:.2f}  "
                f"now ${px:.2f}  {arrow}{pnl:+,.2f} ({pct:+.1f}%)"
            )
        return "\n".join(lines)

    def _trades(args: list[str]) -> str:
        n = 10
        if args and args[0].isdigit():
            n = int(args[0])
        trades = trader.trades[-n:]
        if not trades:
            return "no trades yet"
        lines = [f"Last {len(trades)} trades:"]
        for t in trades:
            lines.append(
                f"{t.timestamp[:19]}  {t.side.upper()} {t.symbol} "
                f"{t.qty:.2f}@${t.price:.2f}"
            )
        return "\n".join(lines)

    def _watchlist(_args: list[str]) -> str:
        try:
            user_extras = sorted(load_user_extras(settings.db_path))
        except Exception:
            user_extras = []
        effective = _merged_symbols(settings, config, wsb_state.extras)
        lines = [f"Effective watchlist ({len(effective)} total):"]
        lines.append(f"📌 Static (config): {', '.join(config.watchlist) or '—'}")
        lines.append(
            f"🔥 WSB-discovered: {', '.join(sorted(wsb_state.extras)) or '—'}"
        )
        lines.append(f"✋ User-added:    {', '.join(user_extras) or '—'}")
        return "\n".join(lines)

    def _add(args: list[str]) -> str:
        if not args:
            return "Usage: /add SYMBOL"
        sym = args[0].upper()
        blocked = set(s.upper() for s in config.watchlist) | set(wsb_state.extras)
        _, msg = add_user_extra(settings.db_path, sym, blocked=blocked)
        return msg

    def _remove(args: list[str]) -> str:
        if not args:
            return "Usage: /remove SYMBOL"
        _, msg = remove_user_extra(settings.db_path, args[0].upper())
        return msg

    def _help(_args: list[str]) -> str:
        return (
            "amms simulate commands:\n"
            "/portfolio — Depot-Übersicht mit P&L\n"
            "/status — Equity + Position-Zahl\n"
            "/positions — alle offenen Trades\n"
            "/trades [N] — letzte N Trades (Standard 10)\n"
            "/watchlist — alle beobachteten Ticker\n"
            "/add SYM — Ticker hinzufügen\n"
            "/remove SYM — Ticker entfernen\n"
            "/help — diese Hilfe"
        )

    handlers = {
        "portfolio": _portfolio,
        "paper": _portfolio,
        "status": _status,
        "positions": _positions,
        "trades": _trades,
        "ptrades": _trades,
        "watchlist": _watchlist,
        "add": _add,
        "remove": _remove,
        "help": _help,
        "start": _help,
    }

    inbound = TelegramInbound(token=token, chat_id=chat_id, handlers=handlers)
    inbound.start()
    logger.info("Telegram inbound started (%d commands)", len(handlers))
    return inbound
