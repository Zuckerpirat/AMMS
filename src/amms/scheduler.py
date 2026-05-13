from __future__ import annotations

import logging
import signal
import sqlite3
import threading
from dataclasses import dataclass, field

from amms import db
from amms.broker import AlpacaClient
from amms.clock import ClockStatus
from amms.config import AppConfig, Settings
from amms.data import MarketDataClient
from amms.executor import TickResult, build_daily_summary, run_tick
from amms.notifier import Notifier, NullNotifier, build_notifier
from amms.strategy import build_strategy

logger = logging.getLogger(__name__)

WHILE_CLOSED_POLL_SECONDS = 60


@dataclass
class LoopState:
    last_saw_open: bool | None = None
    sent_summary_dates: set[str] = field(default_factory=set)


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

    mode = "live (paper)" if execute else "dry-run"
    logger.info("amms scheduler starting in %s mode", mode)
    notifier.send(f"amms started ({mode})")

    state = LoopState()
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
                try:
                    result = run_tick(
                        broker=broker,
                        data=data,
                        conn=conn,
                        config=config,
                        strategy=strategy,
                        bars_back=bars_back,
                        execute=execute,
                    )
                    _announce_tick(notifier, result)
                except Exception:
                    logger.exception("tick failed")
                    notifier.send("amms tick failed; see logs")

                if stop.wait(config.scheduler.tick_seconds):
                    break
    finally:
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
                summary = build_daily_summary(broker, conn)
                notifier.send(summary)
                state.sent_summary_dates.add(today)
            except Exception:
                logger.exception("daily summary failed")
    state.last_saw_open = False


def _announce_tick(notifier: Notifier, result: TickResult) -> None:
    if isinstance(notifier, NullNotifier):
        return
    for order_id in result.placed_order_ids:
        notifier.send(f"amms placed BUY order {order_id}")
