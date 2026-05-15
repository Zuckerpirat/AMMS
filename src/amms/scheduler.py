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
    # Latest macro regime snapshot, refreshed once per tick. The /macro
    # Telegram command reads from here.
    last_macro_regime: object | None = None
    # Dates on which we already sent a drawdown alert; resets daily so
    # the user gets at most one per day.
    drawdown_alert_dates: set[str] = field(default_factory=set)


MAX_CONSECUTIVE_TICK_ERRORS = 5


SENTIMENT_REFRESH_SECONDS = 3600  # hourly

# Mode → RiskConfig field adjustments.
# Conservative: smaller positions, tighter stops.
# Meme: larger positions but still bounded; warn user separately.
# Event: same as swing but with drawdown alert halved.
_MODE_RISK_OVERRIDES: dict[str, dict] = {
    "conservative": {"max_position_pct": 0.01, "stop_loss_pct": 0.03},
    "swing": {},  # default — no adjustment
    "meme": {"max_position_pct": 0.015},
    "event": {"stop_loss_pct": 0.04},
}


def _apply_mode_adjustments(config: AppConfig, mode: str) -> AppConfig:
    """Apply trading-mode-specific risk nudges on top of the effective config."""
    from dataclasses import replace as _replace

    adjustments = _MODE_RISK_OVERRIDES.get(mode, {})
    if not adjustments:
        return config
    try:
        return _replace(config, risk=_replace(config.risk, **adjustments))
    except (TypeError, ValueError):
        return config


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

                def _telegram_backtest(days: int):
                    """Run a backtest using whatever bars are cached locally.

                    Strategy/risk/timeframe come from the live config plus
                    runtime overrides. Initial equity matches the current
                    account so percentage figures are meaningful to the user.
                    """
                    from datetime import date, timedelta
                    from amms.backtest.engine import BacktestConfig, run_backtest
                    from amms.backtest.stats import compute_stats
                    from amms.runtime_overrides import (
                        apply_to_config as _apply,
                        apply_to_strategy as _apply_strat,
                    )

                    cfg = _apply(config, conn)
                    strat = _apply_strat(strategy, conn)
                    try:
                        initial = float(broker.get_account().equity)
                    except Exception:
                        initial = 100_000.0
                    end = date.today()
                    start = end - timedelta(days=days)
                    bt = BacktestConfig(
                        start=start,
                        end=end,
                        symbols=tuple(cfg.watchlist),
                        initial_equity=initial,
                        risk=cfg.risk,
                        strategy=strat,
                        timeframe=cfg.strategy.timeframe,
                        universe=cfg.universe,
                    )
                    result = run_backtest(bt, conn)
                    return compute_stats(result)

                handlers = build_command_handlers(
                    broker=broker,
                    pause=pause,
                    conn=conn,
                    preview=_preview_now,
                    db_path=settings.db_path,
                    static_watchlist=config.watchlist,
                    get_wsb_extras=lambda: set(state.wsb_discovery.extras),
                    data=data,
                    get_macro_regime=lambda: state.last_macro_regime,
                    run_backtest=_telegram_backtest,
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
                # most every refresh_hours; no-op when disabled. Runtime
                # overrides (/set wsb_enabled 1, /set wsb_top_n 10, ...) are
                # applied here so users can toggle expansion from Telegram.
                from amms.runtime_overrides import apply_to_config as _apply_cfg

                pre_overrides_config = _apply_cfg(config, conn)
                effective_config = pre_overrides_config
                if pre_overrides_config.wsb_discovery.enabled:
                    delta = maybe_refresh_wsb_extras(
                        state.wsb_discovery,
                        pre_overrides_config.wsb_discovery,
                        static_watchlist=set(pre_overrides_config.watchlist),
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
                # Apply runtime overrides to the (already extended) config.
                from amms.runtime_overrides import (
                    apply_to_strategy as _apply_strategy_overrides,
                )

                effective_config = _apply_cfg(effective_config, conn)
                effective_strategy = _apply_strategy_overrides(strategy, conn)

                # Trading mode adjustments: map mode → risk parameter nudges.
                # These run after all other overrides so they're always applied.
                from amms.runtime_overrides import get_overrides as _get_overrides

                _ovr_pre = _get_overrides(conn) if conn else {}
                _mode = _ovr_pre.get("trading_mode", "swing")
                effective_config = _apply_mode_adjustments(effective_config, _mode)

                # Macro regime check: when VIXY signals stress, pause buys
                # for this tick so the bot doesn't add long exposure into a
                # vol spike. Stop-loss + sells still flow through normally.
                # Users can tune thresholds or fully disable via /set.
                from amms.data.macro import (
                    DEFAULT_DAY_PCT_STRESS,
                    DEFAULT_WEEK_PCT_STRESS,
                    compute_regime as _compute_regime,
                )
                _ovr = _get_overrides(conn)
                macro_enabled = _ovr.get("macro_enabled", True)
                if macro_enabled:
                    regime = _compute_regime(
                        data,
                        day_pct_stress=_ovr.get(
                            "macro_day_threshold", DEFAULT_DAY_PCT_STRESS
                        ),
                        week_pct_stress=_ovr.get(
                            "macro_week_threshold", DEFAULT_WEEK_PCT_STRESS
                        ),
                    )
                    state.last_macro_regime = regime
                    macro_paused = regime.is_stressed
                    if macro_paused:
                        logger.warning("macro stress detected: %s", regime.reason)
                        notifier.send(f"⚠️ macro-pause: {regime.reason}")
                else:
                    macro_paused = False
                    state.last_macro_regime = None
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

                # Drawdown alert: at most one per day, threshold tunable
                # via /set drawdown_alert (default 5%).
                try:
                    from amms.risk.drawdown import (
                        DEFAULT_DRAWDOWN_ALERT_PCT,
                        compute_drawdown,
                        should_alert as _dd_should_alert,
                    )

                    acc = broker.get_account()
                    dd = compute_drawdown(conn, float(acc.equity))
                    threshold = _ovr.get(
                        "drawdown_alert", DEFAULT_DRAWDOWN_ALERT_PCT
                    )
                    today_iso = clock.timestamp.date().isoformat()
                    if (
                        _dd_should_alert(dd, threshold_pct=threshold)
                        and today_iso not in state.drawdown_alert_dates
                    ):
                        notifier.send(
                            f"🚨 drawdown alert: equity ${dd.current_equity:,.2f} "
                            f"is {dd.drawdown_pct:+.2f}% below 30d peak "
                            f"${dd.peak_equity:,.2f} (threshold {-threshold:+.1f}%)"
                        )
                        state.drawdown_alert_dates.add(today_iso)
                    # Hard stop: auto-pause when drawdown exceeds 2× alert threshold
                    hard_stop_pct = threshold * 2
                    if (
                        _dd_should_alert(dd, threshold_pct=hard_stop_pct)
                        and not pause.paused
                    ):
                        pause.set_paused(True)
                        notifier.send(
                            f"🛑 auto-paused: drawdown {dd.drawdown_pct:+.2f}% "
                            f"exceeds hard-stop threshold {-hard_stop_pct:.1f}%. "
                            f"Use /resume after reviewing."
                        )
                except Exception:
                    logger.exception("drawdown alert check failed")

                # Price alerts: check each tick, fire once per alert.
                try:
                    from amms.data.alerts import check_alerts

                    watchlist_syms = list(
                        {p.symbol for p in broker.get_positions()}
                    )
                    if watchlist_syms:
                        snaps = data.get_snapshots(watchlist_syms)
                        price_map = {
                            s: v["price"]
                            for s, v in snaps.items()
                            if v.get("price")
                        }
                        fired = check_alerts(conn, price_map)
                        for a in fired:
                            notifier.send(
                                f"🔔 price alert: {a.symbol} "
                                f"{'≥' if a.direction == 'above' else '≤'} "
                                f"${a.price:.2f} (now ${price_map.get(a.symbol, 0):.2f})"
                            )
                except Exception:
                    logger.exception("price alert check failed")

                try:
                    result = run_tick(
                        broker=broker,
                        data=data,
                        conn=conn,
                        config=effective_config,
                        strategy=effective_strategy,
                        bars_back=bars_back,
                        execute=execute,
                        paused=pause.paused or macro_paused,
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
                extra = _build_close_extras(broker, conn)
                if extra:
                    plain = plain + "\n" + extra
                trades_today = _fetch_today_trades(conn, today)
                summary = augment_summary(plain, trades_today=trades_today, conn=conn)
                notifier.send(summary)
                state.sent_summary_dates.add(today)
            except Exception:
                logger.exception("daily summary failed")
    state.last_saw_open = False


def _build_close_extras(broker: AlpacaClient, conn: sqlite3.Connection) -> str:
    """Append sector exposure, Sharpe stub, and streak to the end-of-day summary."""
    import math
    from collections import deque

    lines: list[str] = []

    # Sector breakdown of open positions
    try:
        from amms.data.sectors import group_by_sector

        positions = broker.get_positions()
        if positions:
            equity = float(broker.get_account().equity)
            pairs = [(p.symbol, float(getattr(p, "market_value", 0) or 0)) for p in positions]
            sectors = group_by_sector(pairs)
            if equity > 0 and sectors:
                lines.append("Sector exposure:")
                for sec, val in sorted(sectors.items(), key=lambda x: -x[1]):
                    lines.append(f"  {sec}: {val/equity:.1%}")
    except Exception:
        pass

    # Rolling Sharpe (7-day)
    try:
        rows = conn.execute(
            "SELECT substr(ts, 1, 10) AS day, MAX(equity) AS equity "
            "FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) >= date('now', '-7 day') "
            "GROUP BY day ORDER BY day"
        ).fetchall()
        if len(rows) >= 3:
            equities = [float(r[1]) for r in rows]
            rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities))]
            n = len(rets)
            mean = sum(rets) / n
            variance = sum((r - mean) ** 2 for r in rets) / n
            std = math.sqrt(variance) if variance > 0 else 0.0
            sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
            lines.append(f"7d Sharpe: {sharpe:.2f}")
    except Exception:
        pass

    # Win streak
    try:
        order_rows = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price "
            "FROM orders WHERE status = 'filled' "
            "ORDER BY symbol, submitted_at"
        ).fetchall()
        buys: dict[str, deque] = {}
        results: list[bool] = []
        for r in order_rows:
            sym = r["symbol"]
            side = r["side"]
            price = r["filled_avg_price"]
            if price is None:
                continue
            price = float(price)
            if side == "buy":
                if sym not in buys:
                    buys[sym] = deque()
                buys[sym].append(price)
            elif side == "sell" and sym in buys and buys[sym]:
                buy_price = buys[sym].popleft()
                results.append(price > buy_price)
        if results:
            streak_type = "W" if results[-1] else "L"
            streak = 0
            for r in reversed(results):
                if (r and streak_type == "W") or (not r and streak_type == "L"):
                    streak += 1
                else:
                    break
            wins = sum(results)
            total = len(results)
            lines.append(
                f"Streak: {streak}× {streak_type}  |  "
                f"Win rate: {wins}/{total} ({wins/total:.0%})"
            )
    except Exception:
        pass

    return "\n".join(lines)


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
