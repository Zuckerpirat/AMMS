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
    get_macro_regime: Callable[[], object] | None = None,
    run_backtest: Callable[[int], object] | None = None,
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

    # Shared across all handlers; built once per scheduler launch.
    from amms.data.isin import IsinLookup as _IsinLookup
    _isin_cache = _IsinLookup()

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
        isins: dict[str, str] = {}
        try:
            isins = _isin_cache.lookup([p.symbol for p in positions])
        except Exception:
            pass
        from amms.data.sectors import sector_for, UNCLASSIFIED

        # Map symbol -> earliest filled buy date so we can show holding days.
        ages: dict[str, str] = {}
        if conn is not None:
            try:
                rows = conn.execute(
                    "SELECT symbol, MIN(COALESCE(filled_at, submitted_at)) AS first_buy "
                    "FROM orders WHERE side = 'buy' AND status IN ('filled', 'partially_filled') "
                    "GROUP BY symbol"
                ).fetchall()
                for r in rows:
                    if r["first_buy"]:
                        ages[r["symbol"]] = r["first_buy"][:10]
            except Exception:
                pass

        lines: list[str] = []
        today = date.today()
        for p in positions:
            line = (
                f"{p.symbol}: {p.qty:g} @ ${p.avg_entry_price:.2f} "
                f"(mv ${p.market_value:.2f}, P&L ${p.unrealized_pl:+.2f})"
            )
            if p.symbol in ages:
                try:
                    held = (today - date.fromisoformat(ages[p.symbol])).days
                    line += f"  {held}d"
                except Exception:
                    pass
            sector = sector_for(p.symbol)
            if sector != UNCLASSIFIED:
                line += f"  [{sector}]"
            if isins.get(p.symbol):
                line += f"  ISIN {isins[p.symbol]}"
            lines.append(line)
        return "\n".join(lines)

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
            "SELECT submitted_at, symbol, side, qty, status, filled_avg_price "
            "FROM orders ORDER BY submitted_at DESC LIMIT 10"
        ).fetchall()
        if not rows:
            return "no orders yet"
        isins: dict[str, str] = {}
        try:
            isins = _isin_cache.lookup({r["symbol"] for r in rows})
        except Exception:
            pass
        lines: list[str] = []
        for r in rows:
            line = (
                f"{r['submitted_at'][:16]} {r['side'].upper()} {r['qty']:g} "
                f"{r['symbol']} [{r['status']}]"
            )
            price = r["filled_avg_price"]
            if price is not None:
                try:
                    notional = float(r["qty"]) * float(price)
                    line += f" @ ${float(price):.2f} = ${notional:,.2f}"
                except (TypeError, ValueError):
                    pass
            if isins.get(r["symbol"]):
                line += f"  ISIN {isins[r['symbol']]}"
            lines.append(line)
        return "\n".join(lines)

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
        daily_equities: list[float] = []

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
                daily_equities.append(float(last["equity"]))

        if found == 0:
            return "No equity data yet — wait for the first trading tick."

        lines.append(f"Total: ${total_pnl:+.2f}")
        try:
            acc = broker.get_account()
            lines.append(f"Current equity: ${acc.equity:,.2f}")
        except Exception:
            pass
        # ASCII sparkline of the equity curve (oldest → newest).
        if len(daily_equities) >= 2:
            sparkline = _render_sparkline(list(reversed(daily_equities)))
            lines.append(f"Trend: {sparkline}")
        return "\n".join(lines)

    def _render_sparkline(values: list[float]) -> str:
        """Render a list of floats as an 8-level Unicode block sparkline."""
        if not values:
            return ""
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return "▄" * len(values)
        blocks = "▁▂▃▄▅▆▇█"
        out = []
        for v in values:
            idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
            out.append(blocks[idx])
        return "".join(out)

    def _set(args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        from amms.runtime_overrides import allowed_keys, set_override

        if len(args) < 2:
            keys = "\n".join(f"  {k} — {desc}" for k, desc in allowed_keys().items())
            return f"usage: /set KEY VALUE\nAllowed keys:\n{keys}"
        key = args[0].lower()
        raw_value = args[1]
        try:
            value = set_override(conn, key, raw_value)
        except ValueError as e:
            return f"/set rejected: {e}"
        return f"set {key} = {value} (applies on next tick)"

    def _unset(args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        from amms.runtime_overrides import unset_override

        if not args:
            return "usage: /unset KEY"
        key = args[0].lower()
        try:
            removed = unset_override(conn, key)
        except ValueError as e:
            return f"/unset rejected: {e}"
        return f"{key} removed" if removed else f"{key} was not set"

    def _show(_args: list[str]) -> str:
        if conn is None:
            return "DB not wired."
        from amms.runtime_overrides import get_overrides

        overrides = get_overrides(conn)
        if not overrides:
            return "no runtime overrides set (using config.yaml as-is)"
        lines = ["Active overrides:"]
        for k, v in sorted(overrides.items()):
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)

    def _sectors(_args: list[str]) -> str:
        """Show portfolio exposure broken down by sector."""
        positions = broker.get_positions()
        if not positions:
            return "no open positions"
        from amms.data.sectors import group_by_sector

        total_mv = sum(float(p.market_value) for p in positions) or 1.0
        grouped = group_by_sector(
            (p.symbol, float(p.market_value)) for p in positions
        )
        # Sort by value descending.
        ordered = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"Sector exposure (total ${total_mv:,.2f}):"]
        for sector, value in ordered:
            pct = value / total_mv * 100
            lines.append(f"  {sector}: ${value:,.2f} ({pct:.1f}%)")
        return "\n".join(lines)

    def _risk(_args: list[str]) -> str:
        """Show the currently active risk + override settings."""
        try:
            from amms.runtime_overrides import get_overrides
            if conn is not None:
                overrides = get_overrides(conn)
            else:
                overrides = {}
        except Exception:
            overrides = {}
        lines = ["Active risk settings:"]
        defaults = {
            "stop_loss": "0 (disabled)",
            "trailing_stop": "0 (disabled)",
            "max_buys": "config default",
            "drawdown_alert": "5.0%",
            "macro_enabled": "True (default)",
            "macro_day_threshold": "5.0%",
            "macro_week_threshold": "15.0%",
            "sentiment_weight": "config default",
            "wsb_enabled": "config default",
            "wsb_top_n": "config default",
            "wsb_min_mentions": "config default",
        }
        for key, default in defaults.items():
            if key in overrides:
                lines.append(f"  {key}: {overrides[key]}  (override)")
            else:
                lines.append(f"  {key}: {default}")
        return "\n".join(lines)

    def _stops(_args: list[str]) -> str:
        """Show, per position, the distance to its stop-loss trigger."""
        positions = broker.get_positions()
        if not positions:
            return "no open positions"
        # Effective stop-loss percentage from overrides (or unknown).
        stop_pct: float | None = None
        try:
            from amms.runtime_overrides import get_overrides
            if conn is not None:
                stop_pct = get_overrides(conn).get("stop_loss")
        except Exception:
            pass
        if not stop_pct:
            return (
                "Stop-loss not active. Enable with /set stop_loss 0.05 "
                "(5% per-position cap)."
            )
        lines = [f"Stop-loss watch ({stop_pct * 100:.1f}% cap):"]
        for p in positions:
            entry = float(p.avg_entry_price)
            # Use market_value/qty as the live price proxy (broker returns
            # both, and their ratio is the current mark).
            mv = float(p.market_value)
            qty = float(p.qty)
            if qty <= 0:
                continue
            current = mv / qty
            pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0.0
            trigger = entry * (1 - stop_pct)
            distance_pct = (current - trigger) / current * 100 if current else 0.0
            icon = "🟢" if pnl_pct > -stop_pct * 100 * 0.5 else (
                "🟡" if pnl_pct > -stop_pct * 100 * 0.85 else "🔴"
            )
            lines.append(
                f"  {icon} {p.symbol}: ${current:.2f} (P&L {pnl_pct:+.2f}%), "
                f"trigger ${trigger:.2f} ({distance_pct:.2f}% above)"
            )
        return "\n".join(lines)

    def _backtest(args: list[str]) -> str:
        """Run a backtest of the current strategy on the watchlist using
        whatever bars are stored locally. Days default to 90."""
        if run_backtest is None:
            return "backtest not wired (only available when launched from scheduler)"
        try:
            days = int(args[0]) if args else 90
        except ValueError:
            return "usage: /backtest [DAYS]  (default 90, max 730)"
        days = max(5, min(days, 730))
        try:
            stats = run_backtest(days)
        except Exception as e:
            return f"backtest failed: {e!r}"
        if stats is None:
            return "backtest produced no result (no bars in DB?)"
        try:
            ret = stats.total_return_pct
            arrow = "▲" if ret >= 0 else "▼"
            lines = [
                f"Backtest ({days}d): {arrow} {ret:+.2f}%",
                f"  Initial equity: ${stats.initial_equity:,.0f}",
                f"  Final equity:   ${stats.final_equity:,.0f}",
                f"  Trades: {stats.num_trades} "
                f"({stats.num_buys} buys / {stats.num_sells} sells)",
                f"  Closed round-trips: {stats.closed_round_trips}",
                f"  Win rate: {stats.win_rate * 100:.1f}%",
                f"  Max drawdown: {stats.max_drawdown_pct * 100:+.2f}%",
            ]
            return "\n".join(lines)
        except AttributeError:
            return f"backtest result: {stats}"

    def _macro(_args: list[str]) -> str:
        """Show the latest macro regime classification (VIX-proxy based)."""
        regime = None
        if get_macro_regime is not None:
            try:
                regime = get_macro_regime()
            except Exception:
                regime = None
        if regime is None and data is not None:
            try:
                from amms.data.macro import compute_regime
                regime = compute_regime(data)
            except Exception:
                regime = None
        if regime is None:
            return "macro regime not available (need market data)"
        emoji = {"calm": "✅", "elevated": "⚠️", "stressed": "🚨"}.get(
            getattr(regime, "level", ""), ""
        )
        return (
            f"{emoji} Market regime: {regime.level.upper()}\n"
            f"VIXY 1d: {regime.vixy_1d_pct:+.2f}%\n"
            f"VIXY 1w: {regime.vixy_1w_pct:+.2f}%\n"
            f"{regime.reason}"
        )

    def _yesterday(_args: list[str]) -> str:
        """Summarise what happened on the previous calendar day."""
        if conn is None:
            return "DB not wired."
        target = (date.today() - timedelta(days=1)).isoformat()
        first = conn.execute(
            "SELECT equity FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) = ? ORDER BY ts LIMIT 1",
            (target,),
        ).fetchone()
        last = conn.execute(
            "SELECT equity FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) = ? ORDER BY ts DESC LIMIT 1",
            (target,),
        ).fetchone()
        lines = [f"Yesterday ({target}):"]
        if first and last and first["equity"]:
            pnl = last["equity"] - first["equity"]
            pct = pnl / first["equity"] * 100
            arrow = "▲" if pnl >= 0 else "▼"
            lines.append(f"P&L: {arrow} ${pnl:+.2f} ({pct:+.2f}%)")
        else:
            lines.append("P&L: no equity data")
        orders = conn.execute(
            "SELECT symbol, side, qty, status, filled_avg_price FROM orders "
            "WHERE substr(submitted_at, 1, 10) = ? ORDER BY submitted_at",
            (target,),
        ).fetchall()
        if orders:
            lines.append(f"Trades: {len(orders)}")
            for r in orders[:5]:
                line = (
                    f"  {r['side'].upper()} {r['qty']:g} {r['symbol']} "
                    f"[{r['status']}]"
                )
                price = r["filled_avg_price"]
                if price is not None:
                    try:
                        line += f" @ ${float(price):.2f}"
                    except (TypeError, ValueError):
                        pass
                lines.append(line)
            if len(orders) > 5:
                lines.append(f"  … and {len(orders) - 5} more")
        else:
            lines.append("Trades: none")
        return "\n".join(lines)

    def _week(_args: list[str]) -> str:
        """Show last-7-day P&L + trade count summary."""
        if conn is None:
            return "DB not wired."
        end = date.today()
        start = end - timedelta(days=7)
        first = conn.execute(
            "SELECT equity FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) >= ? ORDER BY ts LIMIT 1",
            (start.isoformat(),),
        ).fetchone()
        last = conn.execute(
            "SELECT equity FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) <= ? ORDER BY ts DESC LIMIT 1",
            (end.isoformat(),),
        ).fetchone()
        lines = ["7-day rollup:"]
        if first and last and first["equity"]:
            pnl = last["equity"] - first["equity"]
            pct = pnl / first["equity"] * 100
            arrow = "▲" if pnl >= 0 else "▼"
            lines.append(f"P&L: {arrow} ${pnl:+.2f} ({pct:+.2f}%)")
            lines.append(f"Start equity: ${first['equity']:,.2f}")
            lines.append(f"End equity:   ${last['equity']:,.2f}")
        else:
            lines.append("P&L: no equity data")
        order_count = conn.execute(
            "SELECT COUNT(*) AS c FROM orders "
            "WHERE substr(submitted_at, 1, 10) >= ?",
            (start.isoformat(),),
        ).fetchone()
        buys = conn.execute(
            "SELECT COUNT(*) AS c FROM orders "
            "WHERE side = 'buy' AND substr(submitted_at, 1, 10) >= ?",
            (start.isoformat(),),
        ).fetchone()
        sells = conn.execute(
            "SELECT COUNT(*) AS c FROM orders "
            "WHERE side = 'sell' AND substr(submitted_at, 1, 10) >= ?",
            (start.isoformat(),),
        ).fetchone()
        if order_count and order_count["c"]:
            lines.append(
                f"Trades: {order_count['c']} "
                f"({buys['c']} buys / {sells['c']} sells)"
            )
        else:
            lines.append("Trades: none")
        return "\n".join(lines)

    def _explain(args: list[str]) -> str:
        """Show why the bot's most recent decision on a ticker came out
        the way it did, plus recent signal history. Reads from the
        ``signals`` table — every tick's decision per symbol is persisted
        with its full reason string."""
        if conn is None:
            return "DB not wired."
        if not args:
            return "usage: /explain SYMBOL  (e.g. /explain NVDA)"
        sym = args[0].upper()
        row = conn.execute(
            "SELECT ts, signal, reason, score, strategy FROM signals "
            "WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (sym,),
        ).fetchone()
        if not row:
            return f"no decision recorded yet for {sym}"
        lines = [
            f"Last decision for {sym}:",
            f"  time: {row['ts'][:16]} UTC",
            f"  signal: {row['signal'].upper()}",
            f"  strategy: {row['strategy']}",
        ]
        if row["score"] is not None:
            lines.append(f"  score: {row['score']:+.2f}")
        if row["reason"]:
            lines.append(f"  reason: {row['reason']}")
        # Sentiment overlay context.
        try:
            from amms.strategy.composite import get_sentiment_overlay

            overlay = get_sentiment_overlay()
            if sym in overlay:
                lines.append(
                    f"  WSB attention overlay: {overlay[sym]:+.2f} (0=none, 1=hot)"
                )
        except Exception:
            pass
        try:
            isin = _isin_cache.lookup([sym]).get(sym, "")
            if isin:
                lines.append(f"  ISIN: {isin}")
        except Exception:
            pass

        # Most recent feature snapshot, if present. The features table
        # stores per-symbol indicator values keyed by tick timestamp.
        try:
            feat = conn.execute(
                "SELECT ts, momentum, rsi, realized_vol, rvol "
                "FROM features WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if feat:
                parts = []
                if feat["momentum"] is not None:
                    parts.append(f"momentum {feat['momentum']:+.2%}")
                if feat["rsi"] is not None:
                    parts.append(f"RSI {feat['rsi']:.1f}")
                if feat["realized_vol"] is not None:
                    parts.append(f"vol {feat['realized_vol']:.2%}")
                if feat["rvol"] is not None:
                    parts.append(f"rvol {feat['rvol']:.2f}")
                if parts:
                    lines.append(f"  features: {', '.join(parts)}")
        except Exception:
            pass

        # Recent decision trail (skip the one we already printed).
        try:
            history = conn.execute(
                "SELECT ts, signal, score FROM signals "
                "WHERE symbol = ? AND ts < ? "
                "ORDER BY ts DESC LIMIT 4",
                (sym, row["ts"]),
            ).fetchall()
            if history:
                lines.append("Recent trail:")
                for h in history:
                    s_score = (
                        f" ({h['score']:+.2f})" if h["score"] is not None else ""
                    )
                    lines.append(
                        f"  {h['ts'][:16]} {h['signal'].upper()}{s_score}"
                    )
        except Exception:
            pass

        return "\n".join(lines)

    def _today(_args: list[str]) -> str:
        """One-shot daily snapshot: equity change, trades, positions, trends."""
        today_iso = date.today().isoformat()
        lines: list[str] = [f"Daily snapshot — {today_iso}"]

        # 1. Equity change today.
        if conn is not None:
            first = conn.execute(
                "SELECT equity FROM equity_snapshots "
                "WHERE substr(ts, 1, 10) = ? ORDER BY ts LIMIT 1",
                (today_iso,),
            ).fetchone()
            last = conn.execute(
                "SELECT equity FROM equity_snapshots "
                "WHERE substr(ts, 1, 10) = ? ORDER BY ts DESC LIMIT 1",
                (today_iso,),
            ).fetchone()
            if first and last and first["equity"]:
                pnl = last["equity"] - first["equity"]
                pct = pnl / first["equity"] * 100
                arrow = "▲" if pnl >= 0 else "▼"
                lines.append(
                    f"P&L today: {arrow} ${pnl:+.2f} ({pct:+.2f}%)"
                )
            else:
                lines.append("P&L today: no equity data yet")
        try:
            acc = broker.get_account()
            lines.append(f"Equity: ${acc.equity:,.2f}  Cash: ${acc.cash:,.2f}")
        except Exception:
            pass

        # 2. Trades today.
        if conn is not None:
            order_rows = conn.execute(
                "SELECT symbol, side, qty, status, filled_avg_price FROM orders "
                "WHERE substr(submitted_at, 1, 10) = ? "
                "ORDER BY submitted_at DESC",
                (today_iso,),
            ).fetchall()
            if order_rows:
                order_isins: dict[str, str] = {}
                try:
                    order_isins = _isin_cache.lookup({r["symbol"] for r in order_rows})
                except Exception:
                    pass
                lines.append(f"Trades today: {len(order_rows)}")
                for r in order_rows[:5]:
                    line = (
                        f"  {r['side'].upper()} {r['qty']:g} {r['symbol']} "
                        f"[{r['status']}]"
                    )
                    price = r["filled_avg_price"]
                    if price is not None:
                        try:
                            notional = float(r["qty"]) * float(price)
                            line += f" @ ${float(price):.2f} = ${notional:,.2f}"
                        except (TypeError, ValueError):
                            pass
                    if order_isins.get(r["symbol"]):
                        line += f"  ISIN {order_isins[r['symbol']]}"
                    lines.append(line)
                if len(order_rows) > 5:
                    lines.append(f"  … and {len(order_rows) - 5} more")
            else:
                lines.append("Trades today: none")

        # 3. Open positions with P&L.
        try:
            positions = broker.get_positions()
        except Exception:
            positions = []
        if positions:
            pos_isins: dict[str, str] = {}
            try:
                pos_isins = _isin_cache.lookup([p.symbol for p in positions])
            except Exception:
                pass
            lines.append(f"Open positions: {len(positions)}")
            for p in positions[:5]:
                line = (
                    f"  {p.symbol}: {p.qty:g} @ ${p.avg_entry_price:.2f} "
                    f"(P&L ${p.unrealized_pl:+.2f})"
                )
                if pos_isins.get(p.symbol):
                    line += f"  ISIN {pos_isins[p.symbol]}"
                lines.append(line)
            if len(positions) > 5:
                lines.append(f"  … and {len(positions) - 5} more")
        else:
            lines.append("Open positions: none")

        # 4. Top WSB trending right now (best-effort).
        try:
            from amms.data.wsb_scanner import WSBScanner

            with WSBScanner() as scanner:
                trending = scanner.scan(min_mentions=5, top_n=5)
        except Exception:
            trending = []
        if trending:
            trend_isins: dict[str, str] = {}
            try:
                trend_isins = _isin_cache.lookup([t.symbol for t in trending])
            except Exception:
                pass
            lines.append("WSB trending now:")
            for t in trending:
                line = f"  {t.symbol} ({t.mentions}x)"
                if trend_isins.get(t.symbol):
                    line += f"  ISIN {trend_isins[t.symbol]}"
                lines.append(line)

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

    def _pnl(args: list[str]) -> str:
        """Detailed P&L for one or all positions."""
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        if args:
            sym = args[0].upper()
            positions = [p for p in positions if p.symbol == sym]
            if not positions:
                return f"no open position for {sym}"

        # Fetch current prices for accurate unrealized P&L display.
        price_map: dict[str, float] = {}
        if data is not None:
            try:
                syms = [p.symbol for p in positions]
                snaps = data.get_snapshots(syms)
                price_map = {s: v["price"] for s, v in snaps.items() if v.get("price")}
            except Exception:
                pass

        lines: list[str] = ["Position P&L detail:"]
        total_unrealized = 0.0
        for p in positions:
            current = price_map.get(p.symbol, p.current_price if hasattr(p, "current_price") else None)
            market_value = float(p.qty) * current if current else None
            cost_basis = float(p.qty) * float(p.avg_entry_price)
            unrealized = float(p.unrealized_pl)
            total_unrealized += unrealized
            pct = unrealized / cost_basis * 100 if cost_basis else 0.0
            arrow = "▲" if unrealized >= 0 else "▼"
            line = (
                f"  {p.symbol}: {p.qty:g} × ${float(p.avg_entry_price):.2f} "
                f"= ${cost_basis:,.2f} cost"
            )
            lines.append(line)
            if current:
                lines.append(
                    f"    Now: ${current:.2f}  MktVal: ${market_value:,.2f}  "
                    f"{arrow} {pct:+.2f}% (${unrealized:+.2f})"
                )
            else:
                lines.append(f"    Unrealized: {arrow} ${unrealized:+.2f} ({pct:+.2f}%)")
        if len(positions) > 1:
            total_arrow = "▲" if total_unrealized >= 0 else "▼"
            lines.append(f"Total unrealized: {total_arrow} ${total_unrealized:+.2f}")
        return "\n".join(lines)

    def _mode(args: list[str]) -> str:
        """Show or set the active trading mode."""
        valid_modes = ("conservative", "swing", "meme", "event")
        mode_desc = {
            "conservative": "Long-term research, fundamentals, low volatility, defensive",
            "swing": "Momentum trades, technical analysis, medium-term holds",
            "meme": "Meme stocks, retail hype, social spikes — high risk, sandboxed",
            "event": "Earnings, Fed, macro shocks, hedging, sector rotation",
        }
        if conn is None:
            return "DB not wired — cannot read or set mode."
        if not args:
            from amms.runtime_overrides import get_overrides
            overrides = get_overrides(conn)
            current = overrides.get("trading_mode", "swing")
            lines = [f"Active mode: {current}  — {mode_desc.get(current, '')}"]
            lines.append("\nAvailable modes:")
            for m in valid_modes:
                marker = "→" if m == current else " "
                lines.append(f"  {marker} {m} — {mode_desc[m]}")
            lines.append("\nTo switch: /mode swing")
            return "\n".join(lines)

        new_mode = args[0].lower()
        if new_mode not in valid_modes:
            return f"Unknown mode '{new_mode}'. Choose: {', '.join(valid_modes)}"
        from amms.runtime_overrides import set_override
        try:
            set_override(conn, "trading_mode", new_mode)
        except ValueError as e:
            return f"Error: {e}"
        return f"Trading mode set to: {new_mode}\n{mode_desc[new_mode]}"

    def _alert(args: list[str]) -> str:
        """Manage price alerts.

        Usage:
          /alert AAPL 200 above   — alert when AAPL >= $200
          /alert AAPL 150 below   — alert when AAPL <= $150
          /alert list             — show active alerts
          /alert del 3            — delete alert #3
        """
        if conn is None:
            return "DB not wired — cannot manage alerts."
        from amms.data.alerts import add_alert, delete_alert, list_alerts

        if not args or args[0].lower() == "list":
            alerts = list_alerts(conn)
            if not alerts:
                return "No active price alerts."
            lines = ["Active price alerts:"]
            for a in alerts:
                lines.append(
                    f"  #{a.id} {a.symbol} {a.direction} ${a.price:.2f}"
                )
            return "\n".join(lines)

        if args[0].lower() in ("del", "delete", "rm") and len(args) >= 2:
            try:
                alert_id = int(args[1])
            except ValueError:
                return "usage: /alert del ID"
            deleted = delete_alert(conn, alert_id)
            return f"Alert #{alert_id} deleted." if deleted else f"Alert #{alert_id} not found."

        # Add alert: /alert SYM PRICE above|below
        if len(args) < 3:
            return (
                "usage:\n"
                "  /alert AAPL 200 above   — alert when AAPL >= $200\n"
                "  /alert AAPL 150 below   — alert when AAPL <= $150\n"
                "  /alert list             — show active alerts\n"
                "  /alert del ID           — delete alert"
            )
        sym = args[0].upper()
        try:
            price = float(args[1])
        except ValueError:
            return f"Invalid price: {args[1]!r}"
        direction = args[2].lower()
        if direction not in ("above", "below"):
            return "direction must be 'above' or 'below'"
        try:
            alert = add_alert(conn, sym, price, direction)
        except ValueError as e:
            return f"Error: {e}"
        return f"Alert set: #{alert.id} — notify when {sym} goes {direction} ${price:.2f}"

    def _journal(args: list[str]) -> str:
        """Show completed trades (BUY + SELL pairs) with realized P&L.

        Usage: /journal [SYM]
        """
        if conn is None:
            return "DB not wired."

        sym_filter = args[0].upper() if args else None
        query = (
            "SELECT symbol, side, qty, filled_avg_price, submitted_at "
            "FROM orders WHERE status = 'filled' "
            + ("AND symbol = ? " if sym_filter else "")
            + "ORDER BY symbol, submitted_at"
        )
        params = (sym_filter,) if sym_filter else ()
        rows = conn.execute(query, params).fetchall()

        # Pair BUY/SELL by symbol using a FIFO queue.
        from collections import deque

        buys: dict[str, deque] = {}
        pairs: list[dict] = []
        for r in rows:
            sym = r["symbol"] if hasattr(r, "__getitem__") else r[0]
            side = r["side"] if hasattr(r, "__getitem__") else r[1]
            qty = float(r["qty"] if hasattr(r, "__getitem__") else r[2])
            price = r["filled_avg_price"] if hasattr(r, "__getitem__") else r[3]
            ts = r["submitted_at"] if hasattr(r, "__getitem__") else r[4]
            if price is None:
                continue
            price = float(price)

            if side == "buy":
                if sym not in buys:
                    buys[sym] = deque()
                buys[sym].append({"qty": qty, "price": price, "ts": ts})
            elif side == "sell" and sym in buys and buys[sym]:
                buy = buys[sym].popleft()
                realized = (price - buy["price"]) * min(qty, buy["qty"])
                pairs.append({
                    "symbol": sym,
                    "entry_price": buy["price"],
                    "exit_price": price,
                    "qty": min(qty, buy["qty"]),
                    "realized": realized,
                    "entry_ts": buy["ts"][:10],
                    "exit_ts": ts[:10],
                })

        if not pairs:
            msg = "No completed round-trips"
            if sym_filter:
                msg += f" for {sym_filter}"
            return msg + "."

        lines = [f"Trade journal ({len(pairs)} completed trades):"]
        total_realized = 0.0
        for p in pairs[-10:]:  # show last 10
            arrow = "▲" if p["realized"] >= 0 else "▼"
            pct = (p["exit_price"] - p["entry_price"]) / p["entry_price"] * 100
            lines.append(
                f"  {p['symbol']}: BUY ${p['entry_price']:.2f} → "
                f"SELL ${p['exit_price']:.2f} ({pct:+.2f}%) "
                f"{arrow} ${p['realized']:+.2f}  [{p['entry_ts']} → {p['exit_ts']}]"
            )
            total_realized += p["realized"]
        if len(pairs) > 10:
            lines.append(f"  (showing last 10 of {len(pairs)})")
        lines.append(f"Total realized P&L: ${total_realized:+.2f}")
        return "\n".join(lines)

    def _top(_args: list[str]) -> str:
        """Show best and worst performing open positions by unrealized P&L %."""
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        scored: list[tuple[float, object]] = []
        for p in positions:
            cost = float(p.qty) * float(p.avg_entry_price)
            pct = float(p.unrealized_pl) / cost * 100 if cost else 0.0
            scored.append((pct, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        lines: list[str] = [f"Performance snapshot ({len(scored)} positions):"]
        lines.append("Best:")
        for pct, p in scored[:3]:
            arrow = "▲" if pct >= 0 else "▼"
            lines.append(
                f"  {arrow} {p.symbol}: {pct:+.2f}%  (${float(p.unrealized_pl):+.2f})"
            )
        if len(scored) > 3:
            lines.append("Worst:")
            for pct, p in scored[-3:]:
                arrow = "▲" if pct >= 0 else "▼"
                lines.append(
                    f"  {arrow} {p.symbol}: {pct:+.2f}%  (${float(p.unrealized_pl):+.2f})"
                )
        return "\n".join(lines)

    def _news(args: list[str]) -> str:
        """Show recent news headlines for one or more symbols.

        Usage: /news [SYM ...]
        If no symbol given, uses open positions.
        """
        if data is None:
            return "Market data client not wired."

        if args:
            syms = [a.upper() for a in args[:5]]
        else:
            try:
                positions = broker.get_positions()
                syms = [p.symbol for p in positions[:5]]
            except Exception:
                syms = []
        if not syms:
            return "No symbols to fetch news for. Try /news AAPL NVDA"

        articles = data.get_news(syms, limit=8)
        if not articles:
            return f"No recent news found for {', '.join(syms)}."

        lines = [f"News for {', '.join(syms)}:"]
        for a in articles[:6]:
            headline = a.get("headline", "").strip()
            created = str(a.get("created_at", ""))[:10]
            url = a.get("url", "")
            article_syms = a.get("symbols", [])
            tag = f"[{', '.join(article_syms[:2])}]" if article_syms else ""
            lines.append(f"  {created} {tag} {headline}")
            if url:
                lines.append(f"    {url}")
        return "\n".join(lines)

    def _summary(_args: list[str]) -> str:
        """On-demand AI narrative of the current portfolio state."""
        if conn is None:
            return "DB not wired."
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        # Build a plain-text digest of the current state.
        lines: list[str] = []
        try:
            acc = broker.get_account()
            lines.append(f"Equity: ${acc.equity:,.2f}  Cash: ${acc.cash:,.2f}")
        except Exception:
            lines.append("(equity unavailable)")

        try:
            positions = broker.get_positions()
        except Exception:
            positions = []
        if positions:
            from amms.data.sectors import sector_for

            lines.append(f"Open positions: {len(positions)}")
            for p in positions:
                sector = sector_for(p.symbol)
                lines.append(
                    f"  {p.symbol} ({sector}): {p.qty:g} × ${float(p.avg_entry_price):.2f}"
                    f" | P&L ${float(p.unrealized_pl):+.2f}"
                )
        else:
            lines.append("No open positions.")

        # Recent trades (last 5)
        today_orders = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price, status "
            "FROM orders ORDER BY submitted_at DESC LIMIT 5"
        ).fetchall()
        trades_dicts: list[dict] = []
        for r in today_orders:
            d = dict(r) if hasattr(r, "keys") else {
                "symbol": r[0], "side": r[1], "qty": r[2],
                "filled_avg_price": r[3], "status": r[4],
            }
            trades_dicts.append(d)
        if trades_dicts:
            lines.append("Recent trades:")
            for t in trades_dicts:
                price_str = f" @ ${t['filled_avg_price']:.2f}" if t.get("filled_avg_price") else ""
                lines.append(
                    f"  {t['side'].upper()} {t['qty']:g} {t['symbol']}{price_str} [{t['status']}]"
                )

        plain = "\n".join(lines)

        if not api_key:
            return plain + "\n\n(LLM narration unavailable — set ANTHROPIC_API_KEY)"

        from amms.notifier.llm_summary import augment_summary

        try:
            narrated = augment_summary(
                plain,
                trades_today=trades_dicts,
                conn=conn,
            )
        except Exception as e:
            return plain + f"\n\n(LLM error: {e!r})"
        return narrated

    def _help(_args: list[str]) -> str:
        return (
            "/status — equity + positions + flags\n"
            "/positions — list open positions\n"
            "/equity — just the equity number\n"
            "/today — one-shot daily snapshot (P&L, trades, positions, WSB)\n"
            "/yesterday — yesterday's P&L and trades\n"
            "/week — last-7-day P&L rollup\n"
            "/explain SYM — show why the bot's last decision on SYM was made\n"
            "/macro — current market stress regime (calm/elevated/stressed)\n"
            "/backtest [DAYS] — run a quick backtest of the current strategy\n"
            "/risk — show currently active risk settings + overrides\n"
            "/stops — show distance to stop-loss trigger per position\n"
            "/sectors — portfolio exposure broken down by sector\n"
            "/pnl [SYM] — detailed P&L per position (cost basis, current price, %)\n"
            "/mode [MODE] — show or switch trading mode (conservative/swing/meme/event)\n"
            "/alert SYM PRICE above|below — set a price alert\n"
            "/alert list — show active price alerts\n"
            "/alert del ID — delete a price alert\n"
            "/set KEY VALUE — change a safe runtime setting (stop_loss, trailing_stop, max_buys)\n"
            "/unset KEY — remove a runtime override\n"
            "/show — list active runtime overrides\n"
            "/performance [N] — P&L last N days (default 7)\n"
            "/watchlist — show static + WSB + user-added tickers\n"
            "/add SYM — add ticker to dynamic watchlist\n"
            "/remove SYM — remove ticker from dynamic watchlist\n"
            "/signals — last 10 strategy signals\n"
            "/lastorders — last 10 orders\n"
            "/scan — run WSB Auto-Discovery now\n"
            "/isin SYM — look up the ISIN for a ticker (debug)\n"
            "/buylist — preview what the bot would buy/sell right now\n"
            "/journal [SYM] — completed trade pairs (BUY→SELL) with realized P&L\n"
            "/top — best and worst open positions by unrealized P&L %%\n"
            "/news [SYM] — recent news headlines for a ticker (or open positions)\n"
            "/summary — AI-generated narrative of the current portfolio state\n"
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
        "today": _today,
        "explain": _explain,
        "yesterday": _yesterday,
        "week": _week,
        "macro": _macro,
        "backtest": _backtest,
        "risk": _risk,
        "stops": _stops,
        "sectors": _sectors,
        "set": _set,
        "unset": _unset,
        "show": _show,
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
        "pnl": _pnl,
        "mode": _mode,
        "alert": _alert,
        "summary": _summary,
        "top": _top,
        "news": _news,
        "journal": _journal,
    }
