"""Inbound Telegram command poller.

Long-poll Telegram for messages and dispatch a small set of commands. Runs
in a daemon thread alongside the scheduler. No persistence, no auth beyond
chat_id whitelist (only messages from the configured chat_id are honored).

Supported commands:
    /status       account equity + open positions
    /positions    open positions only
    /equity       just the equity number
    /pause        scheduler stops placing new orders (state flag)
    /resume       scheduler resumes placing orders
    /help         list commands
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

CommandHandler = Callable[[], str]


@dataclass
class PauseFlag:
    """Shared flag flipped by /pause and /resume; checked by executor caller."""

    paused: bool = False

    def set_paused(self, value: bool) -> None:
        self.paused = value


class TelegramInbound:
    """Long-polls Telegram for messages from the configured chat_id."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        handlers: dict[str, CommandHandler],
        *,
        timeout: float = 30.0,
        send_url: str = "https://api.telegram.org",
    ) -> None:
        self._token = token
        self._chat_id = str(chat_id)
        self._handlers = handlers
        self._timeout = timeout
        self._base = f"{send_url}/bot{token}"
        self._last_update_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="amms-telegram-inbound", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.warning("telegram poll failed", exc_info=True)
                # Sleep briefly then retry; don't crash the bot loop.
                self._stop.wait(5)

    def _poll_once(self) -> None:
        params: dict[str, object] = {
            "timeout": int(self._timeout),
            "offset": self._last_update_id + 1 if self._last_update_id else 0,
        }
        try:
            resp = httpx.get(
                f"{self._base}/getUpdates",
                params=params,
                timeout=self._timeout + 5,
            )
            resp.raise_for_status()
        except httpx.RequestError:
            self._stop.wait(2)
            return
        for update in resp.json().get("result", []):
            self._last_update_id = max(self._last_update_id, int(update.get("update_id", 0)))
            self._dispatch(update)

    def _dispatch(self, update: dict) -> None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        chat = message.get("chat", {})
        if str(chat.get("id")) != self._chat_id:
            logger.info("telegram message from wrong chat ignored: %s", chat.get("id"))
            return
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        cmd = text.split()[0].lower().lstrip("/")
        # Strip any @bot suffix Telegram appends in groups (/status@mybot).
        cmd = cmd.split("@", 1)[0]
        handler = self._handlers.get(cmd)
        if handler is None:
            self._reply(f"unknown command: /{cmd}")
            return
        try:
            reply = handler()
        except Exception as e:
            reply = f"command failed: {e!r}"
            logger.exception("telegram handler crashed for /%s", cmd)
        if reply:
            self._reply(reply)

    def _reply(self, text: str) -> None:
        try:
            httpx.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text},
                timeout=self._timeout,
            ).raise_for_status()
        except Exception:
            logger.warning("telegram reply failed", exc_info=True)


def build_command_handlers(
    *,
    broker,
    pause: PauseFlag,
) -> dict[str, CommandHandler]:
    """Construct the default command handler table."""

    def _status() -> str:
        acc = broker.get_account()
        positions = broker.get_positions()
        lines = [
            f"equity: ${acc.equity:,.2f}",
            f"cash: ${acc.cash:,.2f}",
            f"open positions: {len(positions)}",
            f"daytrade_count: {acc.daytrade_count}",
            f"paused: {pause.paused}",
        ]
        return "\n".join(lines)

    def _positions() -> str:
        positions = broker.get_positions()
        if not positions:
            return "no open positions"
        return "\n".join(
            f"{p.symbol}: {p.qty:g} @ ${p.avg_entry_price:.2f} "
            f"(mv ${p.market_value:.2f}, P&L ${p.unrealized_pl:+.2f})"
            for p in positions
        )

    def _equity() -> str:
        return f"${broker.get_account().equity:,.2f}"

    def _pause() -> str:
        pause.set_paused(True)
        return "paused: scheduler will skip placing orders"

    def _resume() -> str:
        pause.set_paused(False)
        return "resumed: scheduler may place orders again"

    def _help() -> str:
        return (
            "/status — equity + positions + flags\n"
            "/positions — list open positions\n"
            "/equity — just the equity number\n"
            "/pause — stop placing new orders\n"
            "/resume — re-enable placing orders\n"
            "/help — this message"
        )

    return {
        "status": _status,
        "positions": _positions,
        "equity": _equity,
        "pause": _pause,
        "resume": _resume,
        "help": _help,
        "start": _help,
    }
