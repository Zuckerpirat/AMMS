from __future__ import annotations

import logging
import os
import signal
import sqlite3
import threading
from dataclasses import dataclass, field

from amms import db
from amms.broker import AlpacaClient
from amms.clock import ClockStatus
from amms.config import AppConfig, Settings
from amms.data import MarketDataClient
from amms.data.wsb_discovery import (
    DiscoveryState,
    format_delta_message,
)
from amms.data.wsb_discovery import (
    maybe_refresh as maybe_refresh_wsb_extras,
)
from amms.executor import TickResult, build_daily_summary, run_tick
from amms.features.sentiment import RedditSentimentCollector
from amms.metrics import start_metrics_server
from amms.notifier import (
    Notifier,
    NullNotifier,
    PauseFlag,
    TelegramInbound,
    build_command_handlers,
    build_notifier,
)
from amms.notifier.llm_summary import augment_summary
from amms.strategy import build_strategy
from amms.strategy.composite import set_sentiment_overlay

logger = logging.getLogger(__name__)

WHILE_CLOSED_POLL_SECONDS = 60


@dataclass
class LoopState:
    last_saw_open: bool | None = None
    sent_summary_dates: set[str] = field(default_factory=set)
    last_sentiment_refresh_ts: float = 0.0
    consecutive_tick_errors: int = 0
    wsb_discovery: DiscoveryState = field(default_factory=DiscoveryState)


MAX_CONSECUTIVE_TICK_ERRORS = 5


SENTIMENT_REFRESH_SECONDS = 3600  # hourly


def _maybe_refresh_sentiment(
    state: LoopState, watchlist: set[str], now_seconds: float
) -> None:
    """Refresh the sentiment overlay from WSB data.

    Uses Reddit's text-scraping path when REDDIT_CLIENT_ID is configured
    (richer per-post sentiment scoring). Otherwise falls back to
    ApeWisdom, where score is derived from raw mention rank — high
    mention counts produce a positive 0..1 overlay value, surfacing
    retail attention to the strategy without needing API keys.
    """
    if now_seconds - state.last_sentiment_refresh_ts < SENTIMENT_REFRESH_SECONDS:
        return
    state.last_sentiment_refresh_ts = now_seconds
    if os.environ.get("REDDIT_CLIENT_ID", "").strip():
        try:
            with RedditSentimentCollector() as coll:
                agg = coll.aggregate(watchlist=watchlist)
            scores = {sym: r.avg_score for sym, r in agg.items()}
            set_sentiment_overlay(scores)
            logger.info("refreshed sentiment overlay (reddit): %d symbols", len(scores))
            return
        except Exception:
            logger.warning("reddit sentiment refresh failed", exc_info=True)
    # Fallback: ApeWisdom mentions → log-normalized score in [0, 1].
    try:
        from amms.features.sentiment import ApeWisdomCollector
        import math

        with ApeWisdomCollector() as ape:
            raw = ape.fetch_trending(filter="wallstreetbets")
        scores: dict[str, float] = {}
        for item in raw:
            sym = (item.get("ticker") or "").upper()
            if not sym or (watchlist and sym not in watchlist):
                continue
            mentions = int(item.get("mentions") or 0)
            if mentions <= 0:
                continue
            # log10/2.5 — caps at 1.0 around 300 mentions so highly-discussed
            # tickers (500+) get the full bonus instead of half.
            scores[sym] = min(1.0, math.log10(mentions + 1) / 2.5)
        set_sentiment_overlay(scores)
        logger.info("refreshed sentiment overlay (apewisdom): %d symbols", len(scores))
    except Exception:
        logger.warning("apewisdom sentiment refresh failed", exc_info=True)


def _install_signal_handlers(stop: threading.Event) -> None:
    def handler(signum: int, _frame: object) -> None:
        logger.info("signal %s received; stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run_loop(
    settings: Settings,
    config: AppConfig,
    *,
    execute: bool = False,
    notifier: Notifier | None = None,
    stop: threading.Event | None = None,
    install_signal_handlers: bool = True,
    bars_back: int = 90,
) -> None:
    """Drive the bot. Ticks during US market hours; idles otherwise.

    `execute=False` is a dry run that produces signals but no orders. The hard
    paper-only guarantee still applies via the broker base-URL check.
    """
    stop = stop if stop is not None else threading.Event()
    if install_signal_handlers:
        _install_signal_handlers(stop)

    notifier = notifier if notifier is not None else build_notifier()
    strategy = build_strategy(config.strategy.name, config.strategy.params)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    pause = PauseFlag()

    mode = "live (paper)" if execute else "dry-run"
    logger.info("amms scheduler starting in %s mode", mode)
    notifier.send(f"amms started ({mode})")

    state = LoopState()
    inbound: TelegramInbound | None = None
    metrics_server = None
    metrics_port_raw = os.environ.get("AMMS_METRICS_PORT", "").strip()
    if metrics_port_raw:
        try:
            metrics_server = start_metrics_server(port=int(metrics_port_raw))
        except Exception:
            logger.exception("failed to start metrics server")
    try:
        with (
            AlpacaClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_base_url,
            ) as broker,
            MarketDataClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_data_url,
            ) as data,
        ):
            # Start inbound Telegram command poller if configured.
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
            if token and chat_id:
                # Closure that runs an immediate read-only tick. Used by
                # /buylist so the user can preview decisions on demand
                # without waiting for the next scheduled tick.
                def _preview_now() -> TickResult:
                    from amms.data.dynamic_watchlist import load as _load
                    from amms.runtime_overrides import (
                        apply_to_config as _apply,
                        apply_to_strategy as _apply_strat,
                    )

                    user_extras = _load(settings.db_path)
                    extras = frozenset(state.wsb_discovery.extras) | frozenset(user_extras)
                    cfg = _with_extra_watchlist(config, extras) if extras else config
                    cfg = _apply(cfg, conn)
                    strat = _apply_strat(strategy, conn)
                    return run_tick(
                        broker=broker,
                        data=data,
                        conn=conn,
                        config=cfg,
                        strategy=strat,
                        bars_back=bars_back,
                        execute=False,
                        paused=pause.paused,
                    )

                handlers = build_command_handlers(
                    broker=broker,
                    pause=pause,
                    conn=conn,
                    preview=_preview_now,
                    db_path=settings.db_path,
                    static_watchlist=config.watchlist,
                    get_wsb_extras=lambda: set(state.wsb_discovery.extras),
                    data=data,
                )
                inbound = TelegramInbound(token=token, chat_id=chat_id, handlers=handlers)
                inbound.start()

            while not stop.is_set():
                try:
                    clock = broker.get_clock()
                except Exception:
                    logger.exception("get_clock failed; retrying after backoff")
                    stop.wait(30)
                    continue

                if not clock.is_open:
                    _handle_closed(broker, conn, notifier, clock, state)
                    if stop.wait(WHILE_CLOSED_POLL_SECONDS):
                        break
                    continue

                state.last_saw_open = True
                import time as _time

                _maybe_refresh_sentiment(
                    state, set(config.watchlist), _time.time()
                )

                # WSB Auto-Discovery: opportunistic watchlist expansion. Runs at
                # most every refresh_hours; no-op when disabled or Reddit creds
                # are missing. We rebuild the per-tick AppConfig with the
                # extended watchlist below.
                effective_config = config
                if config.wsb_discovery.enabled:
                    delta = maybe_refresh_wsb_extras(
                        state.wsb_discovery,
                        config.wsb_discovery,
                        static_watchlist=set(config.watchlist),
                        now_seconds=_time.time(),
                    )
                    if delta.refreshed and (delta.added or delta.removed):
                        notifier.send(format_delta_message(delta))
                # Merge in: WSB-discovered + user-added (via /add). Both
                # layers are recomputed each tick so /add takes effect on the
                # very next tick without a restart.
                from amms.data.dynamic_watchlist import load as _load_user_extras

                user_extras = _load_user_extras(settings.db_path)
                merged_extras = frozenset(state.wsb_discovery.extras) | frozenset(user_extras)
                if merged_extras:
                    effective_config = _with_extra_watchlist(config, merged_extras)
                # Apply any runtime overrides the user set via /set on Telegram.
                from amms.runtime_overrides import (
                    apply_to_config as _apply_overrides,
                    apply_to_strategy as _apply_strategy_overrides,
                )

                effective_config = _apply_overrides(effective_config, conn)
                effective_strategy = _apply_strategy_overrides(strategy, conn)
                # Force-close window: flatten remaining positions just before
                # market close if config.risk.force_close_minutes_before_close
                # is set. Day-trade profiles use this to avoid overnight risk.
                if _in_force_close_window(clock, config.risk.force_close_minutes_before_close):
                    try:
                        ids = _force_close_all(broker, conn, execute=execute)
                        for oid in ids:
                            notifier.send(f"amms force-close placed sell {oid}")
                    except Exception:
                        logger.exception("force_close failed")

                try:
                    result = run_tick(
                        broker=broker,
                        data=data,
                        conn=conn,
                        config=effective_config,
                        strategy=effective_strategy,
                        bars_back=bars_back,
                        execute=execute,
                        paused=pause.paused,
                    )
                    _announce_tick(
                        notifier,
                        result,
                        mode=config.scheduler.tick_notify,
                        execute=execute,
                    )
                    state.consecutive_tick_errors = 0
                except Exception:
                    logger.exception("tick failed")
                    state.consecutive_tick_errors += 1
                    notifier.send(
                        f"amms tick failed "
                        f"({state.consecutive_tick_errors}/{MAX_CONSECUTIVE_TICK_ERRORS})"
                    )
                    if state.consecutive_tick_errors >= MAX_CONSECUTIVE_TICK_ERRORS:
                        pause.set_paused(True)
                        notifier.send(
                            "amms auto-paused after repeated tick errors. "
                            "Use /resume after investigating."
                        )

                if stop.wait(config.scheduler.tick_seconds):
                    break
    finally:
        if inbound is not None:
            inbound.stop()
        if metrics_server is not None:
            metrics_server.shutdown()
        conn.close()
        notifier.send("amms stopped")
        logger.info("amms scheduler stopped")


def _handle_closed(
    broker: AlpacaClient,
    conn: sqlite3.Connection,
    notifier: Notifier,
    clock: ClockStatus,
    state: LoopState,
) -> None:
    if state.last_saw_open is True:
        today = clock.timestamp.date().isoformat()
        if today not in state.sent_summary_dates:
            try:
                plain = build_daily_summary(broker, conn)
                trades_today = _fetch_today_trades(conn, today)
                summary = augment_summary(plain, trades_today=trades_today, conn=conn)
                notifier.send(summary)
                state.sent_summary_dates.add(today)
            except Exception:
                logger.exception("daily summary failed")
    state.last_saw_open = False


def _fetch_today_trades(conn: sqlite3.Connection, today_iso: str) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol, side, qty, status, filled_avg_price "
        "FROM orders WHERE substr(submitted_at, 1, 10) = ? ORDER BY submitted_at",
        (today_iso,),
    ).fetchall()
    return [dict(r) for r in rows]


_MAX_DECISION_LINES = 8  # cap the per-tick digest so Telegram messages stay short


def _announce_tick(
    notifier: Notifier,
    result: TickResult,
    *,
    mode: str = "decisions",
    execute: bool = False,
) -> None:
    """Send a per-tick Telegram digest based on ``mode``.

    Modes:
      - "never": send nothing
      - "orders_only": send only when an order actually went through
      - "decisions": send when there are buy/sell signals OR orders. In
        dry-run this is the most useful — you see what WOULD happen.
      - "always": send every tick, even an empty one
    """
    if isinstance(notifier, NullNotifier) or mode == "never":
        return

    buy_signals = [s for s in result.signals if s.kind == "buy"]
    sell_signals = [s for s in result.signals if s.kind == "sell"]
    has_orders = bool(result.placed_order_ids)
    has_signals = bool(buy_signals or sell_signals)

    if mode == "orders_only" and not has_orders:
        return
    if mode == "decisions" and not (has_orders or has_signals):
        return
    # mode == "always" → fall through and send even an empty tick

    label = "live" if execute else "dry-run"
    header = f"amms tick ({label}): {len(buy_signals)} buy / {len(sell_signals)} sell"
    if has_orders:
        header += f" — placed {len(result.placed_order_ids)} order(s)"

    lines = [header]

    # Show actionable signals first (top by score for buys, all sells).
    actionable = sorted(buy_signals, key=lambda s: s.score, reverse=True) + sell_signals
    for sig in actionable[:_MAX_DECISION_LINES]:
        marker = "BUY " if sig.kind == "buy" else "SELL"
        lines.append(f"{marker} {sig.symbol} @ ${sig.price:.2f} — {sig.reason}")

    # Then show top blocked items so the user understands why something didn't fire.
    if result.blocked:
        remaining = _MAX_DECISION_LINES - (len(actionable) if actionable else 0)
        if remaining > 0:
            for sym, reason in result.blocked[:remaining]:
                lines.append(f"skip {sym}: {reason}")

    if not has_signals and not has_orders:
        lines.append("(no signals; bot is watching)")

    notifier.send("\n".join(lines))


def _in_force_close_window(clock: ClockStatus, minutes_before_close: int) -> bool:
    """True if `minutes_before_close > 0` and we are within that window."""
    if minutes_before_close <= 0:
        return False
    remaining_seconds = (clock.next_close - clock.timestamp).total_seconds()
    return 0 < remaining_seconds <= minutes_before_close * 60


def _force_close_all(
    broker: AlpacaClient,
    conn: sqlite3.Connection,
    *,
    execute: bool,
) -> list[str]:
    """Close every open position via market sell. Returns submitted order IDs."""
    from amms.db import upsert_order

    positions = broker.get_positions()
    open_sells = {
        o.symbol for o in broker.list_orders(status="open") if o.side == "sell"
    }
    placed: list[str] = []
    for pos in positions:
        if pos.symbol in open_sells:
            continue
        if pos.qty <= 0:
            continue
        if not execute:
            logger.info("force-close dry-run: would sell %s %s", pos.qty, pos.symbol)
            continue
        order = broker.submit_order(pos.symbol, pos.qty, "sell")
        upsert_order(conn, order)
        placed.append(order.id)
    return placed


def _with_extra_watchlist(config: AppConfig, extras: frozenset[str]) -> AppConfig:
    """Return a copy of ``config`` whose watchlist has ``extras`` appended.

    Static symbols come first (preserving config-file order); discovered
    extras come last so they are clearly "new" in any debug output.
    Duplicates are collapsed.
    """
    import dataclasses

    static_upper = {s.upper() for s in config.watchlist}
    new_extras = tuple(sorted(s.upper() for s in extras if s.upper() not in static_upper))
    if not new_extras:
        return config
    return dataclasses.replace(
        config, watchlist=tuple(config.watchlist) + new_extras
    )
