"""Inbound Telegram command poller.

Long-poll Telegram for messages and dispatch a small set of commands. Runs
in a daemon thread alongside the scheduler. No persistence, no auth beyond
chat_id whitelist (only messages from the configured chat_id are honored).

Supported commands:
    /status       account equity + open positions
    /positions    open positions only
    /equity       just the equity number
    /scan         run WSB Auto-Discovery now and reply with trending tickers
    /pause        scheduler stops placing new orders (state flag)
    /resume       scheduler resumes placing orders
    /help         list commands
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, timedelta

import httpx

logger = logging.getLogger(__name__)

CommandHandler = Callable[[list[str]], str]
TickPreview = Callable[[], object]  # returns a TickResult


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
        parts = text.split()
        cmd = parts[0].lower().lstrip("/")
        # Strip any @bot suffix Telegram appends in groups (/status@mybot).
        cmd = cmd.split("@", 1)[0]
        args = parts[1:]
        handler = self._handlers.get(cmd)
        if handler is None:
            self._reply(f"unknown command: /{cmd}")
            return
        try:
            reply = handler(args)
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
    conn=None,
    preview: TickPreview | None = None,
    db_path=None,
    static_watchlist: tuple[str, ...] | list[str] = (),
    get_wsb_extras: Callable[[], set[str]] | None = None,
    data=None,
) -> dict[str, CommandHandler]:
    """Construct the default command handler table.

    ``conn`` is an optional sqlite3 connection used by /signals and /lastorders.
    When omitted those commands say "DB not wired" rather than crash.

    ``preview`` is an optional zero-arg callable returning a TickResult
    (typically a closure over the scheduler's broker/data/config/strategy).
    When wired, /buylist runs an immediate read-only tick and reports what
    the bot would do right now without waiting for the next scheduled tick.

    ``db_path`` enables the dynamic-watchlist commands (/add, /remove,
    /watchlist) by pointing them at the persistence file location.
    ``static_watchlist`` and ``get_wsb_extras`` are used to render
    /watchlist with all three layers visible.
    """

    def _status(_args: list[str]) -> str:
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

    def _positions(_args: list[str]) -> str:
        positions = broker.get_positions()
        if not positions:
            return "no open positions"
        return "\n".join(
            f"{p.symbol}: {p.qty:g} @ ${p.avg_entry_price:.2f} "
            f"(mv ${p.market_value:.2f}, P&L ${p.unrealized_pl:+.2f})"
            for p in positions
        )

    def _equity(_args: list[str]) -> str:
        return f"${broker.get_account().equity:,.2f}"

    def _pause(_args: list[str]) -> str:
        pause.set_paused(True)
        return "paused: scheduler will skip placing orders"

    def _resume(_args: list[str]) -> str:
        pause.set_paused(False)
        return "resumed: scheduler may place orders again"

    def _signals(_args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        rows = conn.execute(
            "SELECT ts, symbol, signal, score FROM signals ORDER BY ts DESC LIMIT 10"
        ).fetchall()
        if not rows:
            return "no signals yet"
        return "\n".join(
            f"{r['ts'][:16]} {r['symbol']} {r['signal']}"
            + (f" ({r['score']:.2f})" if r["score"] is not None else "")
            for r in rows
        )

    def _buylist(_args: list[str]) -> str:
        if preview is None:
            return (
                "preview not wired (only available when launched from "
                "the scheduler)."
            )
        try:
            result = preview()
        except Exception as e:
            return f"preview failed: {e!r}"

        buys = sorted(
            [s for s in result.signals if s.kind == "buy"],
            key=lambda s: s.score,
            reverse=True,
        )
        sells = [s for s in result.signals if s.kind == "sell"]

        if not buys and not sells and not result.blocked:
            return "no buy/sell signals right now; bot is watching."

        lines = [f"preview: {len(buys)} buy / {len(sells)} sell signals"]
        for sig in buys[:10]:
            lines.append(
                f"BUY {sig.symbol} @ ${sig.price:.2f} (score {sig.score:+.2f})"
                f" — {sig.reason}"
            )
        for sig in sells[:5]:
            lines.append(
                f"SELL {sig.symbol} @ ${sig.price:.2f} — {sig.reason}"
            )
        if result.blocked:
            lines.append("--- skipped ---")
            for sym, reason in result.blocked[:8]:
                lines.append(f"{sym}: {reason}")
        return "\n".join(lines)

    # Persistent across /scan invocations so OpenFIGI is only queried once
    # per ticker per process lifetime.
    from amms.data.isin import IsinLookup as _IsinLookup
    _isin_cache = _IsinLookup()

    def _scan(_args: list[str]) -> str:
        # Lazy import to avoid circular deps and to keep the inbound module
        # cheap to import (it is loaded even when WSB scanning is unused).
        from amms.data.wsb_scanner import WSBScanner, format_summary

        try:
            with WSBScanner() as scanner:
                results = scanner.scan(
                    limit_per_sub=100, time_filter="day", min_mentions=3, top_n=10
                )
        except Exception as e:
            return f"WSB scan failed: {e!r}"

        prices: dict[str, dict[str, float]] = {}
        isins: dict[str, str] = {}
        if results:
            syms = [r.symbol for r in results]
            if data is not None:
                try:
                    prices = data.get_snapshots(syms)
                except Exception:
                    prices = {}
            try:
                isins = _isin_cache.lookup(syms)
            except Exception:
                isins = {}
        return format_summary(results, prices=prices, isins=isins)

    def _lastorders(_args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        rows = conn.execute(
            "SELECT submitted_at, symbol, side, qty, status "
            "FROM orders ORDER BY submitted_at DESC LIMIT 10"
        ).fetchall()
        if not rows:
            return "no orders yet"
        return "\n".join(
            f"{r['submitted_at'][:16]} {r['side'].upper()} {r['qty']:g} "
            f"{r['symbol']} [{r['status']}]"
            for r in rows
        )

    def _add(args: list[str]) -> str:
        if db_path is None:
            return "watchlist commands not wired (no db_path)."
        if not args:
            return "usage: /add SYMBOL  (e.g. /add NVDA)"
        from amms.data.dynamic_watchlist import add as add_symbol

        try:
            _, msg = add_symbol(db_path, args[0], blocked=set(static_watchlist))
        except ValueError as e:
            return str(e)
        return msg

    def _remove(args: list[str]) -> str:
        if db_path is None:
            return "watchlist commands not wired (no db_path)."
        if not args:
            return "usage: /remove SYMBOL  (e.g. /remove NVDA)"
        from amms.data.dynamic_watchlist import remove as remove_symbol

        try:
            _, msg = remove_symbol(db_path, args[0])
        except ValueError as e:
            return str(e)
        return msg

    def _watchlist(_args: list[str]) -> str:
        if db_path is None:
            return "watchlist commands not wired (no db_path)."
        from amms.data.dynamic_watchlist import format_summary, load

        user_extras = load(db_path)
        wsb_extras = get_wsb_extras() if get_wsb_extras is not None else set()
        return format_summary(static_watchlist, wsb_extras, user_extras)

    def _performance(args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        try:
            days = int(args[0]) if args and args[0].isdigit() else 7
            days = max(1, min(days, 30))
        except (ValueError, IndexError):
            days = 7

        today = date.today()
        lines: list[str] = [f"P&L last {days} days"]
        total_pnl = 0.0
        found = 0

        for i in range(days):
            d = today - timedelta(days=i)
            d_str = d.isoformat()
            first = conn.execute(
                "SELECT equity FROM equity_snapshots "
                "WHERE substr(ts, 1, 10) = ? ORDER BY ts LIMIT 1",
                (d_str,),
            ).fetchone()
            last = conn.execute(
                "SELECT equity FROM equity_snapshots "
                "WHERE substr(ts, 1, 10) = ? ORDER BY ts DESC LIMIT 1",
                (d_str,),
            ).fetchone()
            if first and last and first["equity"]:
                pnl = last["equity"] - first["equity"]
                pct = (pnl / first["equity"]) * 100
                arrow = "▲" if pnl >= 0 else "▼"
                lines.append(f"{d_str}: {arrow} ${pnl:+.2f} ({pct:+.2f}%)")
                total_pnl += pnl
                found += 1

        if found == 0:
            return "No equity data yet — wait for the first trading tick."

        lines.append(f"Total: ${total_pnl:+.2f}")
        try:
            acc = broker.get_account()
            lines.append(f"Current equity: ${acc.equity:,.2f}")
        except Exception:
            pass
        return "\n".join(lines)

    def _isin(args: list[str]) -> str:
        if not args:
            return "usage: /isin SYM [SYM ...]  (e.g. /isin NVDA SPY)"
        syms = [a.upper() for a in args]
        try:
            mapping = _isin_cache.lookup(syms)
        except Exception as e:
            return f"isin lookup failed: {e!r}"
        lines = []
        for sym in syms:
            isin = mapping.get(sym, "")
            lines.append(f"{sym}: {isin or '— not found'}")
        return "\n".join(lines)

    def _help(_args: list[str]) -> str:
        return (
            "/status — equity + positions + flags\n"
            "/positions — list open positions\n"
            "/equity — just the equity number\n"
            "/performance [N] — P&L last N days (default 7)\n"
            "/watchlist — show static + WSB + user-added tickers\n"
            "/add SYM — add ticker to dynamic watchlist\n"
            "/remove SYM — remove ticker from dynamic watchlist\n"
            "/signals — last 10 strategy signals\n"
            "/lastorders — last 10 orders\n"
            "/scan — run WSB Auto-Discovery now\n"
            "/isin SYM — look up the ISIN for a ticker (debug)\n"
            "/buylist — preview what the bot would buy/sell right now\n"
            "/pause — stop placing new orders\n"
            "/resume — re-enable placing orders\n"
            "/help — this message"
        )

    return {
        "status": _status,
        "positions": _positions,
        "equity": _equity,
        "signals": _signals,
        "lastorders": _lastorders,
        "scan": _scan,
        "isin": _isin,
        "wsb": _scan,  # alias
        "buylist": _buylist,
        "preview": _buylist,  # alias
        "performance": _performance,
        "perf": _performance,  # alias
        "add": _add,
        "remove": _remove,
        "watchlist": _watchlist,
        "pause": _pause,
        "resume": _resume,
        "help": _help,
        "start": _help,
    }
