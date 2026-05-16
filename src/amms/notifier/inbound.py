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
    reload_config: "Callable[[], str] | None" = None,
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

    def _riskreport(_args: list[str]) -> str:
        """Comprehensive risk diagnostic: concentration, drawdown, sectors, corr."""
        import math

        sections: list[str] = ["=== Risk Report ==="]

        # 1. Drawdown vs 30d peak
        if conn is not None:
            try:
                from amms.risk.drawdown import compute_drawdown

                acc = broker.get_account()
                dd = compute_drawdown(conn, float(acc.equity))
                arrow = "▼" if dd.drawdown_pct < 0 else "▲"
                sections.append(
                    f"Drawdown: {arrow} {dd.drawdown_pct:.2f}%  "
                    f"(equity ${dd.current_equity:,.0f}, 30d peak ${dd.peak_equity:,.0f})"
                )
            except Exception:
                pass

        # 2. Sector concentration
        try:
            positions = broker.get_positions()
        except Exception:
            positions = []
        if positions:
            try:
                acc = broker.get_account()
                equity = float(acc.equity)
                from amms.data.sectors import group_by_sector

                pairs = [(p.symbol, float(getattr(p, "market_value", 0) or 0)) for p in positions]
                sectors = group_by_sector(pairs)
                sections.append("Sector concentration:")
                for sec, val in sorted(sectors.items(), key=lambda x: -x[1]):
                    pct = val / equity * 100 if equity > 0 else 0
                    flag = " ⚠" if pct > 40 else ""
                    sections.append(f"  {sec}: {pct:.1f}%{flag}")
            except Exception:
                pass

        # 3. Largest single position
        if positions:
            try:
                acc = broker.get_account()
                equity = float(acc.equity)
                biggest = max(positions, key=lambda p: float(getattr(p, "market_value", 0) or 0))
                biggest_pct = float(getattr(biggest, "market_value", 0)) / equity * 100
                flag = " ⚠ (concentrated)" if biggest_pct > 20 else ""
                sections.append(
                    f"Largest position: {biggest.symbol} "
                    f"${float(getattr(biggest, 'market_value', 0)):,.0f}  "
                    f"({biggest_pct:.1f}% of equity){flag}"
                )
            except Exception:
                pass

        # 4. Correlation warning (uses DB bars)
        if conn is not None and len(positions) >= 2:
            try:
                syms = [p.symbol for p in positions]
                closes: dict[str, list[float]] = {}
                for sym in syms:
                    rows = conn.execute(
                        "SELECT close FROM bars WHERE symbol = ? AND timeframe = '1Day' "
                        "ORDER BY ts DESC LIMIT 30",
                        (sym,),
                    ).fetchall()
                    if rows and len(rows) >= 5:
                        closes[sym] = [float(r[0]) for r in reversed(rows)]
                rets: dict[str, list[float]] = {}
                for sym, c in closes.items():
                    rets[sym] = [(c[i] - c[i - 1]) / c[i - 1] for i in range(1, len(c))]
                present = list(closes.keys())
                high_corr_pairs = []
                for i, s1 in enumerate(present):
                    for s2 in present[i + 1:]:
                        a, b = rets[s1], rets[s2]
                        n = min(len(a), len(b))
                        if n < 5:
                            continue
                        a, b = a[:n], b[:n]
                        ma, mb = sum(a) / n, sum(b) / n
                        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
                        sa = math.sqrt(sum((ai - ma) ** 2 for ai in a))
                        sb = math.sqrt(sum((bi - mb) ** 2 for bi in b))
                        if sa > 0 and sb > 0:
                            c_val = num / (sa * sb)
                            if abs(c_val) >= 0.75:
                                high_corr_pairs.append((s1, s2, c_val))
                if high_corr_pairs:
                    sections.append("High correlation pairs (>0.75):")
                    for s1, s2, c_val in high_corr_pairs:
                        sections.append(f"  {s1} ↔ {s2}: {c_val:+.2f} ⚠")
                else:
                    sections.append("Correlation: no highly correlated pairs")
            except Exception:
                pass

        # 5. Open overrides summary
        if conn is not None:
            try:
                from amms.runtime_overrides import get_overrides

                ovr = get_overrides(conn)
                if ovr:
                    sections.append(
                        "Active overrides: "
                        + ", ".join(f"{k}={v}" for k, v in ovr.items())
                    )
            except Exception:
                pass

        return "\n".join(sections)

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
                f"  Max drawdown: {stats.max_drawdown_pct:+.2f}%",
            ]
            # Enhanced stats (may not be present on older BacktestStats)
            if hasattr(stats, "sharpe") and stats.sharpe != 0:
                lines.append(f"  Sharpe (ann.): {stats.sharpe:.2f}")
            if hasattr(stats, "profit_factor") and stats.profit_factor > 0:
                lines.append(f"  Profit factor: {stats.profit_factor:.2f}")
            if hasattr(stats, "avg_win") and stats.avg_win > 0:
                lines.append(
                    f"  Avg win: ${stats.avg_win:.2f}  |  "
                    f"Avg loss: ${getattr(stats, 'avg_loss', 0):.2f}"
                )
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

    def _streak(_args: list[str]) -> str:
        """Show the current win/loss streak from completed round-trip trades."""
        if conn is None:
            return "DB not wired."

        rows = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price, submitted_at "
            "FROM orders WHERE status = 'filled' "
            "ORDER BY symbol, submitted_at"
        ).fetchall()

        from collections import deque

        buys: dict[str, deque] = {}
        results: list[bool] = []  # True = win, False = loss

        for r in rows:
            sym = r["symbol"]
            side = r["side"]
            qty = float(r["qty"])
            price = r["filled_avg_price"]
            if price is None:
                continue
            price = float(price)

            if side == "buy":
                if sym not in buys:
                    buys[sym] = deque()
                buys[sym].append({"qty": qty, "price": price})
            elif side == "sell" and sym in buys and buys[sym]:
                buy = buys[sym].popleft()
                win = price > buy["price"]
                results.append(win)

        if not results:
            return "No completed trades yet."

        total = len(results)
        wins = sum(results)
        losses = total - wins
        win_rate = wins / total * 100

        # Compute current streak (from end of list)
        streak_type = "W" if results[-1] else "L"
        streak = 0
        for r in reversed(results):
            if (r and streak_type == "W") or (not r and streak_type == "L"):
                streak += 1
            else:
                break

        # Longest win / loss streaks
        best_win_streak = 0
        best_loss_streak = 0
        cur_w = 0
        cur_l = 0
        for r in results:
            if r:
                cur_w += 1
                cur_l = 0
                best_win_streak = max(best_win_streak, cur_w)
            else:
                cur_l += 1
                cur_w = 0
                best_loss_streak = max(best_loss_streak, cur_l)

        arrow = "🟢" if streak_type == "W" else "🔴"
        lines = [
            f"Trade streak ({total} completed):",
            f"  Current streak: {arrow} {streak}× {streak_type}",
            f"  Win rate: {wins}W / {losses}L  ({win_rate:.1f}%)",
            f"  Best win streak:  {best_win_streak}",
            f"  Worst loss streak: {best_loss_streak}",
        ]
        return "\n".join(lines)

    def _sharpe(_args: list[str]) -> str:
        """Compute rolling Sharpe ratio from the equity curve (last 30 days)."""
        if conn is None:
            return "DB not wired."

        import math

        rows = conn.execute(
            "SELECT substr(ts, 1, 10) AS day, MAX(equity) AS equity "
            "FROM equity_snapshots "
            "WHERE substr(ts, 1, 10) >= date('now', '-30 day') "
            "GROUP BY day ORDER BY day"
        ).fetchall()

        if len(rows) < 5:
            return "Not enough equity history (need 5+ days)."

        equities = [float(r[1]) for r in rows]
        # Daily returns
        rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities))]
        n = len(rets)
        mean = sum(rets) / n
        variance = sum((r - mean) ** 2 for r in rets) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Annualised Sharpe (252 trading days, assume 0 risk-free rate)
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

        total_return = (equities[-1] - equities[0]) / equities[0] * 100
        max_eq = max(equities)
        min_trough = min(equities[equities.index(max_eq):]) if max_eq in equities else equities[-1]
        max_dd = (min_trough - max_eq) / max_eq * 100 if max_eq > 0 else 0.0

        lines = [
            f"Performance metrics ({n} daily returns):",
            f"  Period return:  {total_return:+.2f}%",
            f"  Daily avg:      {mean*100:+.4f}%",
            f"  Daily vol:      {std*100:.4f}%",
            f"  Sharpe (ann.):  {sharpe:.2f}",
            f"  Max drawdown:   {max_dd:.2f}%",
        ]
        return "\n".join(lines)

    def _sentiment(args: list[str]) -> str:
        """Show current WSB sentiment for a symbol.

        Usage: /sentiment [SYM]  — defaults to open positions if no symbol given
        """
        syms: list[str] = [a.upper() for a in args[:3]] if args else []
        if not syms:
            try:
                positions = broker.get_positions()
                syms = [p.symbol for p in positions[:5]]
            except Exception:
                pass
        if not syms:
            return "No symbols. Try /sentiment AAPL NVDA"

        try:
            from amms.features.sentiment import ApeWisdomCollector

            coll = ApeWisdomCollector()
            trending = coll.fetch_trending(filter="all-stocks", page=1)
        except Exception as e:
            return f"WSB data unavailable: {e!r}"

        by_sym = {item["ticker"].upper(): item for item in trending if item.get("ticker")}
        lines = [f"WSB sentiment for: {', '.join(syms)}"]
        for sym in syms:
            if sym in by_sym:
                item = by_sym[sym]
                rank = item.get("rank", "?")
                mentions = item.get("mentions", 0)
                mention_24h = item.get("mentions_24h", 0)
                lines.append(
                    f"  {sym}: rank #{rank}, {mentions} mentions total, "
                    f"{mention_24h} in last 24h"
                )
            else:
                lines.append(f"  {sym}: not in top trending")
        return "\n".join(lines)

    def _profit(args: list[str]) -> str:
        """Show realized P&L grouped by period.

        Usage: /profit [day|week|month|all]  — default: week
        """
        if conn is None:
            return "DB not wired."

        period = (args[0].lower() if args else "week")
        if period not in ("day", "week", "month", "all"):
            return "usage: /profit [day|week|month|all]"

        if period == "day":
            date_filter = "date('now')"
        elif period == "week":
            date_filter = "date('now', '-7 day')"
        elif period == "month":
            date_filter = "date('now', '-30 day')"
        else:
            date_filter = "'1970-01-01'"

        rows = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price, submitted_at "
            "FROM orders WHERE status = 'filled' "
            f"AND substr(submitted_at, 1, 10) >= {date_filter} "
            "ORDER BY symbol, submitted_at"
        ).fetchall()

        from collections import deque

        buys: dict[str, deque] = {}
        pairs: list[dict] = []
        for r in rows:
            sym = r["symbol"]
            side = r["side"]
            qty = float(r["qty"])
            price = r["filled_avg_price"]
            if price is None:
                continue
            price = float(price)
            if side == "buy":
                if sym not in buys:
                    buys[sym] = deque()
                buys[sym].append({"qty": qty, "price": price})
            elif side == "sell" and sym in buys and buys[sym]:
                buy = buys[sym].popleft()
                realized = (price - buy["price"]) * min(qty, buy["qty"])
                pairs.append({"symbol": sym, "realized": realized})

        if not pairs:
            return f"No completed trades in the selected period ({period})."

        total = sum(p["realized"] for p in pairs)
        wins = sum(1 for p in pairs if p["realized"] > 0)
        losses = len(pairs) - wins

        by_sym: dict[str, float] = {}
        for p in pairs:
            by_sym[p["symbol"]] = by_sym.get(p["symbol"], 0.0) + p["realized"]

        arrow = "▲" if total >= 0 else "▼"
        lines = [
            f"Realized P&L — {period}:",
            f"  Total: {arrow} ${total:+.2f}  ({wins}W / {losses}L)",
            "By symbol:",
        ]
        for sym, pnl in sorted(by_sym.items(), key=lambda x: -abs(x[1])):
            a = "▲" if pnl >= 0 else "▼"
            lines.append(f"  {sym}: {a} ${pnl:+.2f}")
        return "\n".join(lines)

    def _upcoming(_args: list[str]) -> str:
        """Show upcoming earnings dates for held positions (next 14 days)."""
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "No open positions to check earnings for."

        syms = [p.symbol for p in positions]
        try:
            from amms.data.earnings import fetch_upcoming

            events = fetch_upcoming(syms, days_ahead=14)
        except Exception as e:
            return f"Earnings lookup failed: {e!r}"

        if not events:
            return f"No earnings in the next 14 days for: {', '.join(syms)}"

        lines = ["Upcoming earnings (next 14 days):"]
        for ev in sorted(events, key=lambda e: e.date):
            time_tag = f" ({ev.time})" if ev.time else ""
            eps_tag = f"  EPS est: {ev.eps_estimate}" if ev.eps_estimate != "N/A" else ""
            lines.append(f"  {ev.symbol}: {ev.date}{time_tag}{eps_tag}")
        return "\n".join(lines)

    def _compare(args: list[str]) -> str:
        """Compare two tickers side by side.

        Usage: /compare AAPL MSFT
        Shows: sector, price, daily change, weekly change, ATR.
        """
        if data is None:
            return "Market data client not wired."
        if len(args) < 2:
            return "usage: /compare SYM1 SYM2  (e.g. /compare AAPL NVDA)"

        syms = [a.upper() for a in args[:2]]
        try:
            snaps = data.get_snapshots(syms)
        except Exception as e:
            return f"data error: {e!r}"

        from amms.data.sectors import sector_for

        lines = [f"Comparison: {syms[0]} vs {syms[1]}"]
        headers = ["Metric", syms[0], syms[1]]
        rows: list[tuple[str, str, str]] = []

        def _fmt(snap: dict, key: str, fmt: str = ".2f") -> str:
            val = snap.get(key)
            if val is None:
                return "N/A"
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        snap0 = snaps.get(syms[0], {})
        snap1 = snaps.get(syms[1], {})

        rows.append(("Sector", sector_for(syms[0]), sector_for(syms[1])))
        rows.append(("Price", f"${_fmt(snap0, 'price')}", f"${_fmt(snap1, 'price')}"))
        rows.append((
            "Day change",
            f"{_fmt(snap0, 'change_pct', '+.2f')}%",
            f"{_fmt(snap1, 'change_pct', '+.2f')}%",
        ))
        rows.append((
            "Week change",
            f"{_fmt(snap0, 'change_pct_week', '+.2f')}%",
            f"{_fmt(snap1, 'change_pct_week', '+.2f')}%",
        ))

        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        lines.append("  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
        lines.append("  ".join("-" * w for w in col_w))
        for row in rows:
            lines.append("  ".join(str(row[i]).ljust(col_w[i]) for i in range(3)))
        return "\n".join(lines)

    def _budget(_args: list[str]) -> str:
        """Show available buying power and estimated position slots remaining."""
        try:
            acc = broker.get_account()
        except Exception as e:
            return f"broker error: {e!r}"
        try:
            positions = broker.get_positions()
        except Exception:
            positions = []

        equity = float(acc.equity)
        cash = float(acc.cash)

        # Read risk config from overrides if available
        max_pos_pct = 0.02  # default
        max_positions = 5  # default
        if conn is not None:
            try:
                from amms.runtime_overrides import get_overrides
                overrides = get_overrides(conn)
                if "stop_loss" in overrides:
                    pass  # no max_position_pct override yet
            except Exception:
                pass

        max_position_dollars = equity * max_pos_pct
        open_slots = max(0, max_positions - len(positions))
        affordable_slots = int(cash // max_position_dollars) if max_position_dollars > 0 else 0
        usable_slots = min(open_slots, affordable_slots)

        lines = [
            f"Budget summary:",
            f"  Equity:        ${equity:>12,.2f}",
            f"  Cash:          ${cash:>12,.2f}",
            f"  Max per pos:   ${max_position_dollars:>12,.2f}  ({max_pos_pct:.0%} of equity)",
            f"  Open positions: {len(positions)} / {max_positions}",
            f"  Available slots: {open_slots}",
            f"  Affordable slots: {affordable_slots}",
            f"  Usable slots: {usable_slots}",
        ]
        if usable_slots == 0:
            if len(positions) >= max_positions:
                lines.append("→ Position limit reached. Sell to free a slot.")
            else:
                lines.append("→ Insufficient cash for another full position.")
        return "\n".join(lines)

    def _corr(_args: list[str]) -> str:
        """Show pairwise return correlation between held positions (30-day bars)."""
        if conn is None:
            return "DB not wired."

        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if len(positions) < 2:
            return "Need at least 2 open positions to compute correlation."

        syms = [p.symbol for p in positions]

        # Load closes from the DB (last 30 bars per symbol)
        closes: dict[str, list[float]] = {}
        for sym in syms:
            rows = conn.execute(
                "SELECT close FROM bars WHERE symbol = ? AND timeframe = '1Day' "
                "ORDER BY ts DESC LIMIT 30",
                (sym,),
            ).fetchall()
            if rows and len(rows) >= 5:
                closes[sym] = [float(r[0]) for r in reversed(rows)]

        present = [s for s in syms if s in closes]
        if len(present) < 2:
            return "Not enough bar history in DB — wait for a few ticks."

        # Compute daily returns
        returns: dict[str, list[float]] = {}
        for sym in present:
            c = closes[sym]
            returns[sym] = [(c[i] - c[i - 1]) / c[i - 1] for i in range(1, len(c))]

        import math

        def _corr_pair(a: list[float], b: list[float]) -> float:
            n = min(len(a), len(b))
            if n < 2:
                return float("nan")
            a, b = a[:n], b[:n]
            ma = sum(a) / n
            mb = sum(b) / n
            num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
            sa = math.sqrt(sum((ai - ma) ** 2 for ai in a))
            sb = math.sqrt(sum((bi - mb) ** 2 for bi in b))
            if sa == 0 or sb == 0:
                return float("nan")
            return num / (sa * sb)

        lines = [f"Return correlation ({len(present)} positions, ~30-day):"]
        for i, s1 in enumerate(present):
            for s2 in present[i + 1:]:
                c = _corr_pair(returns[s1], returns[s2])
                if math.isnan(c):
                    lines.append(f"  {s1} ↔ {s2}: insufficient data")
                else:
                    level = "high" if abs(c) >= 0.7 else "moderate" if abs(c) >= 0.4 else "low"
                    lines.append(f"  {s1} ↔ {s2}: {c:+.2f}  [{level}]")

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

    def _export(_args: list[str]) -> str:
        """Export trade history as CSV text.

        Usage: /export [N]  — last N filled orders (default 20)
        The output is plain CSV that can be copy-pasted into a spreadsheet.
        """
        if conn is None:
            return "DB not wired."

        limit = 20
        if _args:
            try:
                limit = int(_args[0])
            except ValueError:
                return "usage: /export [N]  (number of orders)"

        rows = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price, status, submitted_at, filled_at "
            "FROM orders WHERE status = 'filled' "
            "ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        if not rows:
            return "No filled orders."

        header = "symbol,side,qty,price,notional,submitted_at,filled_at"
        lines = [header]
        for r in rows:
            qty = float(r["qty"])
            price = float(r["filled_avg_price"] or 0)
            notional = qty * price
            lines.append(
                f"{r['symbol']},{r['side']},{qty:g},{price:.2f},"
                f"{notional:.2f},"
                f"{r['submitted_at'] or ''},"
                f"{r['filled_at'] or ''}"
            )
        return "```\n" + "\n".join(lines) + "\n```"

    def _ping(_args: list[str]) -> str:
        from datetime import UTC, datetime

        ts = datetime.now(UTC).strftime("%H:%M:%S UTC")
        try:
            acc = broker.get_account()
            return f"pong! {ts}  |  equity ${acc.equity:,.2f}"
        except Exception:
            return f"pong! {ts}  (broker unreachable)"

    def _version(_args: list[str]) -> str:
        import subprocess

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            sha = "unknown"
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            branch = "unknown"
        return f"amms  git {sha}  branch {branch}"

    def _fees(args: list[str]) -> str:
        """Estimate cumulative transaction cost for paper trades.

        Alpaca paper trading has no real fees, but this simulates the cost
        using a configurable basis-point rate so the user can see impact.
        Usage: /fees [bps]  — default 5 bps per trade (0.05%)
        """
        if conn is None:
            return "DB not wired."

        bps = 5.0
        if args:
            try:
                bps = float(args[0])
            except ValueError:
                return "usage: /fees [BPS]  (e.g. /fees 5 for 5 basis points)"

        rate = bps / 10_000

        rows = conn.execute(
            "SELECT side, qty, filled_avg_price "
            "FROM orders WHERE status = 'filled' AND filled_avg_price IS NOT NULL"
        ).fetchall()

        if not rows:
            return "No filled orders yet."

        total_notional = sum(
            float(r["qty"]) * float(r["filled_avg_price"]) for r in rows
        )
        total_fees = total_notional * rate
        trade_count = len(rows)

        lines = [
            f"Simulated fee analysis ({bps:.1f} bps/trade):",
            f"  Trades:          {trade_count}",
            f"  Total notional:  ${total_notional:,.2f}",
            f"  Estimated fees:  ${total_fees:,.2f}",
            f"  Avg per trade:   ${total_fees/trade_count:.2f}",
            "",
            "(Alpaca paper trades have no real fees. This is a simulation.)",
        ]
        return "\n".join(lines)

    def _setlist(args: list[str]) -> str:
        """Bulk-replace the entire dynamic watchlist.

        Usage:
          /setlist SYM [SYM ...]   — replace list with given symbols
          /setlist clear           — empty the list (same as /setlist with no args)
        """
        if db_path is None:
            return "watchlist commands not wired (no db_path)."
        from amms.data.dynamic_watchlist import _save_raw, _watchlist_path, normalize_symbol

        if not args or (len(args) == 1 and args[0].lower() == "clear"):
            path = _watchlist_path(db_path)
            _save_raw(path, set())
            return "Dynamic watchlist cleared (0 symbols)."

        new_syms: set[str] = set()
        bad: list[str] = []
        for raw in args:
            try:
                new_syms.add(normalize_symbol(raw))
            except ValueError:
                bad.append(raw)

        if bad:
            return f"Invalid tickers: {', '.join(bad)}. Use 1-5 uppercase letters."

        path = _watchlist_path(db_path)
        _save_raw(path, new_syms)
        return f"Dynamic watchlist replaced with {len(new_syms)} symbol(s): {', '.join(sorted(new_syms))}"

    def _calendar(_args: list[str]) -> str:
        """Show NYSE market hours and upcoming US market holidays."""
        from datetime import UTC, datetime

        # Fixed NYSE 2026 holidays (approximate — observe nearest weekday)
        NYSE_HOLIDAYS_2026 = [
            ("2026-01-01", "New Year's Day"),
            ("2026-01-19", "Martin Luther King Jr. Day"),
            ("2026-02-16", "Presidents' Day"),
            ("2026-04-03", "Good Friday"),
            ("2026-05-25", "Memorial Day"),
            ("2026-07-03", "Independence Day (observed)"),
            ("2026-09-07", "Labor Day"),
            ("2026-11-26", "Thanksgiving Day"),
            ("2026-11-27", "Black Friday (early close 1:00 PM ET)"),
            ("2026-12-25", "Christmas Day"),
        ]

        today = date.today()
        upcoming = [
            (d, name) for d, name in NYSE_HOLIDAYS_2026
            if date.fromisoformat(d) >= today
        ][:5]

        now_utc = datetime.now(UTC)
        # NYSE opens 14:30 UTC, closes 21:00 UTC (ET = UTC-5 in winter, UTC-4 in summer)
        # Simplify: just show ET offset note
        lines = [
            "NYSE Market Hours:",
            "  Monday–Friday   9:30 AM – 4:00 PM ET",
            "  Pre-market:     4:00 AM – 9:30 AM ET",
            "  After-hours:    4:00 PM – 8:00 PM ET",
            "",
            f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC",
        ]

        if upcoming:
            lines += ["", "Upcoming NYSE holidays:"]
            for d, name in upcoming:
                lines.append(f"  {d}  {name}")
        else:
            lines.append("\n(No more NYSE holidays in 2026 data.)")

        return "\n".join(lines)

    def _drawdown(_args: list[str]) -> str:
        """Detailed drawdown analytics: current DD, historical worst, recovery info."""
        if conn is None:
            return "DB not wired."
        try:
            acc = broker.get_account()
            current_equity = float(acc.equity)
        except Exception as e:
            return f"broker error: {e!r}"

        try:
            from amms.risk.drawdown import compute_drawdown
            state = compute_drawdown(conn, current_equity, lookback_days=30)
        except Exception as e:
            return f"drawdown error: {e!r}"

        # Compute historical worst drawdown from full equity_snapshots history
        try:
            rows = conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY ts ASC"
            ).fetchall()
            equities = [float(r[0]) for r in rows if r[0] is not None]
        except Exception:
            equities = []

        worst_dd = 0.0
        recovery_days: int | None = None
        if len(equities) >= 2:
            peak = equities[0]
            in_drawdown = False
            dd_start = 0
            for i, eq in enumerate(equities):
                if eq > peak:
                    if in_drawdown:
                        # recovered
                        if recovery_days is None:
                            recovery_days = i - dd_start
                        in_drawdown = False
                    peak = eq
                dd = (eq - peak) / peak * 100
                if dd < worst_dd:
                    worst_dd = dd
                    if not in_drawdown:
                        in_drawdown = True
                        dd_start = i

        sign = "" if state.drawdown_pct >= 0 else ""
        lines = [
            "Drawdown analytics (30-day window):",
            f"  Peak equity:    ${state.peak_equity:>12,.2f}",
            f"  Current equity: ${state.current_equity:>12,.2f}",
            f"  Current DD:     {state.drawdown_pct:+.2f}%",
            "",
            f"  All-time worst DD: {worst_dd:.2f}%",
        ]
        if recovery_days is not None:
            lines.append(f"  Last recovery:  {recovery_days} day(s)")
        if abs(state.drawdown_pct) > 5.0:
            lines.append("\n⚠️  Drawdown alert threshold exceeded.")
        return "\n".join(lines)

    def _alloc(args: list[str]) -> str:
        """Show portfolio allocation by sector vs a simple equal-weight target.

        Usage: /alloc — show current sector weights vs equal-weight target
        """
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        from amms.data.sectors import group_by_sector

        total_mv = sum(float(p.market_value) for p in positions)
        if total_mv <= 0:
            return "no market value"

        grouped = group_by_sector(
            (p.symbol, float(p.market_value)) for p in positions
        )
        n_sectors = len(grouped)
        target_pct = 100.0 / n_sectors if n_sectors else 0.0

        ordered = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"Allocation vs equal-weight target ({target_pct:.1f}% per sector):"]
        for sector, value in ordered:
            actual_pct = value / total_mv * 100
            diff = actual_pct - target_pct
            arrow = "▲" if diff > 1 else ("▼" if diff < -1 else "≈")
            lines.append(
                f"  {arrow} {sector:<22} actual {actual_pct:5.1f}%  target {target_pct:.1f}%  diff {diff:+.1f}%"
            )
        lines.append(f"\nTotal market value: ${total_mv:,.2f}")
        return "\n".join(lines)

    def _heatmap(_args: list[str]) -> str:
        """Text heat-map: positions ranked by intraday % change with bar chart."""
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        items: list[tuple[str, float, float]] = []
        for p in positions:
            try:
                entry = float(p.avg_entry_price)
                mv = float(p.market_value)
                qty = float(p.qty)
                if qty > 0 and entry > 0:
                    current_price = mv / qty
                    pct = (current_price - entry) / entry * 100
                    items.append((p.symbol, pct, mv))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        if not items:
            return "No position data available."

        items.sort(key=lambda x: x[1], reverse=True)

        max_abs = max(abs(pct) for _, pct, _ in items) or 1.0
        bar_width = 20
        lines = ["Position heat-map (vs entry price):"]
        for sym, pct, mv in items:
            bar_len = int(abs(pct) / max_abs * bar_width)
            bar = ("█" * bar_len).ljust(bar_width)
            sign = "+" if pct >= 0 else "-"
            arrow = "▲" if pct >= 0 else "▼"
            lines.append(
                f"  {arrow} {sym:<6} {sign}{abs(pct):5.2f}%  [{bar}]  ${mv:>10,.0f}"
            )
        return "\n".join(lines)

    def _limit(args: list[str]) -> str:
        """Show or set the maximum number of buy orders per day.

        Usage:
          /limit          — show current daily buy limit
          /limit N        — set daily max buys to N
          /limit off      — remove override (use config default)
        """
        if conn is None:
            return "DB not wired."

        if not args:
            # Show current
            try:
                from amms.runtime_overrides import get_overrides
                overrides = get_overrides(conn)
                current = overrides.get("max_buys", "(config default)")
            except Exception:
                current = "(unknown)"
            return f"Daily buy limit: {current}\nUse /limit N to change, /limit off to remove override."

        val = args[0].lower()
        if val == "off":
            from amms.runtime_overrides import unset_override
            try:
                unset_override(conn, "max_buys")
            except Exception as e:
                return f"error removing override: {e!r}"
            return "Daily buy limit override removed (using config default)."

        try:
            n = int(val)
            if n < 0:
                raise ValueError
        except ValueError:
            return "usage: /limit N  (positive integer) or /limit off"

        from amms.runtime_overrides import set_override
        try:
            set_override(conn, "max_buys", str(n))
        except Exception as e:
            return f"error: {e!r}"
        return f"Daily buy limit set to {n} orders/day."

    def _vol(args: list[str]) -> str:
        """Show 20-day realized volatility and ATR(14) for held positions or a given ticker.

        Usage: /vol [SYM]
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /vol AAPL)"

        from amms.features.volatility import atr as compute_atr, realized_vol

        lines = ["Volatility metrics (20d realized vol, ATR-14):"]
        for sym in symbols[:8]:  # cap at 8 to keep reply short
            try:
                bars = data.get_bars(sym, limit=30)
            except Exception:
                bars = []
            rv = realized_vol(bars, 20)
            a = compute_atr(bars, 14)
            rv_str = f"{rv * 100:.1f}%" if rv is not None else "n/a"
            atr_str = f"${a:.2f}" if a is not None else "n/a"
            lines.append(f"  {sym:<6}  vol {rv_str:<8}  ATR {atr_str}")
        return "\n".join(lines)

    def _reload(_args: list[str]) -> str:
        """Trigger a config reload from disk without restarting the bot.

        The scheduler must wire in a reload_config callback for this to work.
        """
        if reload_config is None:
            return "reload not wired (scheduler doesn't support hot-reload)."
        try:
            result = reload_config()
            return result if isinstance(result, str) else "Config reloaded."
        except Exception as e:
            return f"reload failed: {e!r}"

    def _bench(args: list[str]) -> str:
        """Compare portfolio equity curve vs SPY over N days.

        Usage: /bench [N]  — default 30 days
        Pulls equity_snapshots from DB and SPY bars, computes % return of each.
        """
        if conn is None:
            return "DB not wired."

        days = 30
        if args:
            try:
                days = max(1, min(int(args[0]), 365))
            except ValueError:
                return "usage: /bench [N]  (number of days, e.g. /bench 30)"

        # Portfolio returns from equity_snapshots
        try:
            rows = conn.execute(
                "SELECT equity FROM equity_snapshots "
                "WHERE substr(ts, 1, 10) >= date('now', ?) "
                "ORDER BY ts ASC",
                (f"-{days} day",),
            ).fetchall()
        except Exception as e:
            return f"DB error: {e!r}"

        equities = [float(r[0]) for r in rows if r[0] is not None]
        if len(equities) < 2:
            return f"Not enough equity history for {days}-day benchmark."

        port_return = (equities[-1] - equities[0]) / equities[0] * 100

        # SPY benchmark via data client if available
        spy_return_str = "n/a (data client not wired)"
        if data is not None:
            try:
                spy_bars = data.get_bars("SPY", limit=days + 5)
                if len(spy_bars) >= 2:
                    spy_start = spy_bars[0].close
                    spy_end = spy_bars[-1].close
                    spy_return = (spy_end - spy_start) / spy_start * 100
                    spy_return_str = f"{spy_return:+.2f}%"
            except Exception:
                spy_return_str = "n/a (error fetching SPY)"

        alpha_str = ""
        if spy_return_str not in ("n/a (data client not wired)", "n/a (error fetching SPY)"):
            try:
                spy_ret = float(spy_return_str.strip("%+"))
                alpha = port_return - spy_ret
                alpha_str = f"\n  Alpha (vs SPY):   {alpha:+.2f}%"
            except ValueError:
                pass

        lines = [
            f"Benchmark comparison (last {days} days):",
            f"  Portfolio return: {port_return:+.2f}%",
            f"  SPY return:       {spy_return_str}",
        ]
        if alpha_str:
            lines.append(alpha_str)
        lines.append(f"\n  Based on {len(equities)} equity snapshots.")
        return "\n".join(lines)

    def _targets(_args: list[str]) -> str:
        """Show entry price, stop-loss level, and take-profit target per position.

        Stop-loss derived from runtime override or default 3% below entry.
        Take-profit shown at 2× the risk (2R).
        """
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        stop_pct = 0.03  # default
        if conn is not None:
            try:
                from amms.runtime_overrides import get_overrides
                overrides = get_overrides(conn)
                sl = overrides.get("stop_loss", 0.0)
                if sl and float(sl) > 0:
                    stop_pct = float(sl)
            except Exception:
                pass

        lines = [f"Position targets (stop-loss {stop_pct:.1%} | take-profit 2R):"]
        for p in positions:
            try:
                entry = float(p.avg_entry_price)
                stop = entry * (1 - stop_pct)
                target = entry * (1 + stop_pct * 2)
                mv = float(p.market_value)
                qty = float(p.qty)
                current = mv / qty if qty > 0 else entry
                dist_pct = (current - entry) / entry * 100
                lines.append(
                    f"  {p.symbol:<6}  entry ${entry:.2f}  "
                    f"stop ${stop:.2f}  target ${target:.2f}  "
                    f"({dist_pct:+.1f}% vs entry)"
                )
            except (TypeError, ValueError, ZeroDivisionError):
                lines.append(f"  {p.symbol}: data error")
        return "\n".join(lines)

    def _mdd(_args: list[str]) -> str:
        """Show the 5 worst single-day equity drops from equity_snapshots."""
        if conn is None:
            return "DB not wired."
        try:
            rows = conn.execute(
                "SELECT substr(ts, 1, 10) AS day, equity "
                "FROM equity_snapshots ORDER BY ts ASC"
            ).fetchall()
        except Exception as e:
            return f"DB error: {e!r}"

        if len(rows) < 2:
            return "Not enough equity history for daily drawdown table."

        daily: list[tuple[str, float]] = []
        for i in range(1, len(rows)):
            prev = float(rows[i - 1]["equity"])
            curr = float(rows[i]["equity"])
            if prev > 0:
                chg = (curr - prev) / prev * 100
                daily.append((rows[i]["day"], chg))

        if not daily:
            return "No daily changes found."

        worst = sorted(daily, key=lambda x: x[1])[:5]
        lines = ["5 worst single-day equity drops:"]
        for rank, (day, chg) in enumerate(worst, 1):
            lines.append(f"  #{rank}  {day}  {chg:+.2f}%")
        best = sorted(daily, key=lambda x: x[1], reverse=True)[:3]
        lines.append("\n3 best single-day gains:")
        for rank, (day, chg) in enumerate(best, 1):
            lines.append(f"  #{rank}  {day}  {chg:+.2f}%")
        return "\n".join(lines)

    def _optout(args: list[str]) -> str:
        """Block or unblock a specific ticker from being traded.

        Usage:
          /optout SYM         — block SYM from being bought (still on watchlist)
          /optout SYM remove  — unblock SYM
          /optout list        — show all blocked tickers
        """
        if conn is None:
            return "DB not wired."

        # Persist the blocklist directly in runtime_overrides table (key: blocked_symbols).
        # We bypass get_overrides() because blocked_symbols is not in _ALLOWED.
        try:
            from amms.runtime_overrides import ensure_table
            ensure_table(conn)
            row = conn.execute(
                "SELECT value FROM runtime_overrides WHERE key = 'blocked_symbols'"
            ).fetchone()
        except Exception as e:
            return f"DB error: {e!r}"

        raw_blocked = str(row[0] if row else "").strip()
        blocked: set[str] = {s.strip().upper() for s in raw_blocked.split(",") if s.strip()}

        if not args or args[0].lower() == "list":
            if not blocked:
                return "No tickers blocked. Use /optout SYM to block one."
            return "Blocked from trading:\n" + "\n".join(f"  {s}" for s in sorted(blocked))

        try:
            from amms.data.dynamic_watchlist import normalize_symbol
            sym = normalize_symbol(args[0])
        except ValueError as e:
            return str(e)

        action = args[1].lower() if len(args) > 1 else "add"

        if action in ("remove", "unblock", "del", "delete"):
            if sym not in blocked:
                return f"{sym} is not blocked."
            blocked.discard(sym)
        else:
            blocked.add(sym)

        new_val = ",".join(sorted(blocked))
        conn.execute(
            "INSERT OR REPLACE INTO runtime_overrides (key, value) VALUES (?, ?)",
            ("blocked_symbols", new_val),
        )
        conn.commit()

        if action in ("remove", "unblock", "del", "delete"):
            return f"{sym} unblocked. Blocked list: {', '.join(sorted(blocked)) or '(empty)'}"
        return f"{sym} blocked from trading. Blocked list: {', '.join(sorted(blocked))}"

    def _note(args: list[str]) -> str:
        """Attach or read a freetext note for a ticker.

        Usage:
          /note SYM text...   — save a note for SYM (replaces previous note)
          /note SYM           — read the note for SYM
          /note list          — list all tickers with notes
          /note SYM clear     — delete the note for SYM
        """
        if conn is None:
            return "DB not wired."

        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ticker_notes "
                "(symbol TEXT PRIMARY KEY, note TEXT, updated_at TEXT)"
            )
            conn.commit()
        except Exception as e:
            return f"DB error: {e!r}"

        if not args:
            return "usage: /note SYM [text]  or  /note list"

        if args[0].lower() == "list":
            rows = conn.execute(
                "SELECT symbol, updated_at FROM ticker_notes ORDER BY symbol"
            ).fetchall()
            if not rows:
                return "No notes saved. Use /note SYM text to add one."
            lines = ["Tickers with notes:"]
            for r in rows:
                try:
                    sym_r = r["symbol"]
                    ts_r = r["updated_at"]
                except (KeyError, TypeError):
                    sym_r, ts_r = r[0], r[1]
                lines.append(f"  {sym_r}  (updated {ts_r[:10] if ts_r else '?'})")
            return "\n".join(lines)

        from amms.data.dynamic_watchlist import normalize_symbol
        try:
            sym = normalize_symbol(args[0])
        except ValueError as e:
            return str(e)

        if len(args) == 1:
            # Read mode
            row = conn.execute(
                "SELECT note, updated_at FROM ticker_notes WHERE symbol = ?", (sym,)
            ).fetchone()
            if not row:
                return f"No note for {sym}. Use /note {sym} text to add one."
            try:
                note_text = row["note"]
                ts = row["updated_at"]
            except (KeyError, TypeError):
                note_text, ts = row[0], row[1]
            return f"{sym} ({ts[:10] if ts else '?'}):\n{note_text}"

        if args[1].lower() == "clear":
            conn.execute("DELETE FROM ticker_notes WHERE symbol = ?", (sym,))
            conn.commit()
            return f"Note for {sym} deleted."

        from datetime import UTC, datetime
        note_body = " ".join(args[1:])
        conn.execute(
            "INSERT OR REPLACE INTO ticker_notes (symbol, note, updated_at) VALUES (?, ?, ?)",
            (sym, note_body, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        return f"Note saved for {sym}: {note_body}"

    def _recap(_args: list[str]) -> str:
        """Brief daily recap: equity, trades, P&L, positions, top mover."""
        lines = ["=== Daily Recap ==="]

        # Equity
        try:
            acc = broker.get_account()
            lines.append(f"Equity: ${float(acc.equity):,.2f}  |  Cash: ${float(acc.cash):,.2f}")
        except Exception:
            lines.append("Equity: (error fetching account)")

        # Positions
        try:
            positions = broker.get_positions()
            n = len(positions)
        except Exception:
            positions = []
            n = 0

        if positions:
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            lines.append(f"Open positions: {n}  |  Total unrealized P&L: ${total_pl:+,.2f}")
            # Top mover
            try:
                top = max(
                    positions,
                    key=lambda p: abs(float(p.unrealized_pl) / max(float(p.market_value), 1)),
                )
                top_pct = float(top.unrealized_pl) / max(float(top.market_value), 1) * 100
                lines.append(f"Biggest mover: {top.symbol} {top_pct:+.1f}%")
            except Exception:
                pass
        else:
            lines.append("No open positions.")

        # Today's trades from DB
        if conn is not None:
            try:
                today_str = date.today().isoformat()
                rows = conn.execute(
                    "SELECT side, COUNT(*) AS cnt FROM orders "
                    "WHERE status='filled' AND substr(filled_at,1,10)=? "
                    "GROUP BY side",
                    (today_str,),
                ).fetchall()
                if rows:
                    parts = []
                    for r in rows:
                        try:
                            side, cnt = r["side"], r["cnt"]
                        except (KeyError, TypeError):
                            side, cnt = r[0], r[1]
                        parts.append(f"{cnt}× {side}")
                    lines.append(f"Today's fills: {', '.join(parts)}")
                else:
                    lines.append("Today's fills: none")
            except Exception:
                pass

        # Pause status
        lines.append(f"Bot status: {'⏸ PAUSED' if pause.paused else '▶ running'}")
        return "\n".join(lines)

    def _rsi(args: list[str]) -> str:
        """Show 14-day RSI for open positions or a specified ticker.

        RSI > 70 = overbought, RSI < 30 = oversold.
        Usage: /rsi [SYM]
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /rsi AAPL)"

        from amms.features.momentum import rsi as compute_rsi

        lines = ["RSI-14 indicators:"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=30)
            except Exception:
                bars = []
            r = compute_rsi(bars, 14)
            if r is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
            else:
                label = "🔴 overbought" if r > 70 else ("🟢 oversold" if r < 30 else "⚪ neutral")
                lines.append(f"  {sym:<6}  RSI {r:.1f}  {label}")
        return "\n".join(lines)

    def _ema_cmd(args: list[str]) -> str:
        """Show EMA-20 and EMA-50 crossover status for positions or a ticker.

        Usage: /ema [SYM]
        Bullish: price > EMA-20 > EMA-50. Bearish: price < EMA-20 < EMA-50.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /ema AAPL)"

        from amms.features.momentum import ema as compute_ema

        lines = ["EMA crossover status (EMA-20 / EMA-50):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            e20 = compute_ema(bars, 20)
            e50 = compute_ema(bars, 50)
            price = bars[-1].close if bars else None
            if e20 is None or e50 is None or price is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            if price > e20 > e50:
                signal = "📈 bullish"
            elif price < e20 < e50:
                signal = "📉 bearish"
            else:
                signal = "↔️ mixed"
            lines.append(
                f"  {sym:<6}  ${price:.2f}  EMA20 ${e20:.2f}  EMA50 ${e50:.2f}  {signal}"
            )
        return "\n".join(lines)

    def _macd_cmd(args: list[str]) -> str:
        """Show MACD(12,26,9) for open positions or a specified ticker.

        Positive histogram = bullish momentum. Negative = bearish.
        Usage: /macd [SYM]
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /macd AAPL)"

        from amms.features.momentum import macd as compute_macd

        lines = ["MACD (12,26,9) indicators:"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = compute_macd(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            m_line, sig_line, hist = result
            trend = "📈" if hist > 0 else "📉"
            cross = " 🔔 crossover!" if abs(m_line - sig_line) < abs(hist) * 0.1 else ""
            lines.append(
                f"  {sym:<6}  MACD {m_line:+.4f}  signal {sig_line:+.4f}  "
                f"hist {hist:+.4f} {trend}{cross}"
            )
        return "\n".join(lines)

    def _score(args: list[str]) -> str:
        """Composite signal score [-100..+100] for a ticker or open positions.

        Combines: RSI (contrarian), EMA trend, 20-day momentum, ATR/vol.
        Usage: /score [SYM]
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /score AAPL)"

        from amms.features.momentum import ema as compute_ema, n_day_return, rsi as compute_rsi
        from amms.features.volatility import realized_vol

        lines = ["Composite signal score (-100..+100):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []

            score = 0.0
            components: list[str] = []

            # RSI component (contrarian: oversold=+, overbought=-)
            r = compute_rsi(bars, 14)
            if r is not None:
                rsi_score = (50 - r) * 0.6  # max ±30
                score += rsi_score
                components.append(f"RSI={r:.0f}")

            # EMA trend component
            e20 = compute_ema(bars, 20)
            e50 = compute_ema(bars, 50)
            if e20 and e50:
                if e20 > e50:
                    score += 20
                    components.append("EMA↑")
                else:
                    score -= 20
                    components.append("EMA↓")

            # Momentum (20d return)
            mom = n_day_return(bars, 20)
            if mom is not None:
                score += mom * 500  # 10% move → ±50 pts, capped later
                components.append(f"mom={mom*100:.1f}%")

            # Volatility penalty (high vol = lower confidence)
            rv = realized_vol(bars, 20)
            if rv and rv > 0.4:
                score -= 10
                components.append("hiVol")

            score = max(-100, min(100, score))
            bar_w = int(abs(score) / 2)
            bar = ("█" * bar_w).ljust(50)
            sign = "+" if score >= 0 else ""
            lines.append(
                f"  {sym:<6}  score {sign}{score:.0f}  [{bar[:20]}]  ({', '.join(components)})"
            )
        return "\n".join(lines)

    def _filter(args: list[str]) -> str:
        """Screen the active watchlist by signal criteria.

        Usage:
          /filter                   — score >= 0 (positive signals only)
          /filter score N           — score >= N (e.g. /filter score 30)
          /filter rsi oversold      — RSI < 35
          /filter rsi overbought    — RSI > 65
          /filter bull              — EMA-20 > EMA-50 (bullish trend)
          /filter bear              — EMA-20 < EMA-50 (bearish trend)
        """
        if data is None:
            return "Data client not wired."

        # Determine the active watchlist
        symbols: list[str] = list(static_watchlist)
        if get_wsb_extras is not None:
            symbols += list(get_wsb_extras())
        if db_path is not None:
            from amms.data.dynamic_watchlist import load as load_dyn
            symbols += list(load_dyn(db_path))
        symbols = list(dict.fromkeys(s.upper() for s in symbols))  # dedupe

        if not symbols:
            return "Watchlist is empty. Add tickers with /add SYM."

        # Parse filter criteria
        mode = "score"
        threshold = 0.0
        if args:
            if args[0].lower() == "score":
                mode = "score"
                threshold = float(args[1]) if len(args) > 1 else 0.0
            elif args[0].lower() == "rsi":
                mode = "rsi_" + (args[1].lower() if len(args) > 1 else "oversold")
            elif args[0].lower() in ("bull", "bullish"):
                mode = "bull"
            elif args[0].lower() in ("bear", "bearish"):
                mode = "bear"

        from amms.features.momentum import ema as compute_ema, n_day_return, rsi as compute_rsi
        from amms.features.volatility import realized_vol

        matches: list[tuple[str, float, str]] = []
        for sym in symbols[:30]:  # cap scan to 30 tickers
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            if not bars:
                continue

            if mode == "score" or mode.startswith("rsi"):
                r = compute_rsi(bars, 14)
            else:
                r = None

            if mode.startswith("rsi"):
                if r is None:
                    continue
                if mode == "rsi_oversold" and r >= 35:
                    continue
                if mode == "rsi_overbought" and r <= 65:
                    continue
                matches.append((sym, r, f"RSI {r:.1f}"))
                continue

            if mode in ("bull", "bear"):
                e20 = compute_ema(bars, 20)
                e50 = compute_ema(bars, 50)
                if e20 is None or e50 is None:
                    continue
                if mode == "bull" and not (e20 > e50):
                    continue
                if mode == "bear" and not (e20 < e50):
                    continue
                matches.append((sym, e20 - e50, f"EMA20-EMA50={e20-e50:+.2f}"))
                continue

            # Score mode
            score = 0.0
            if r is not None:
                score += (50 - r) * 0.6
            e20 = compute_ema(bars, 20)
            e50 = compute_ema(bars, 50)
            if e20 and e50:
                score += 20 if e20 > e50 else -20
            mom = n_day_return(bars, 20)
            if mom is not None:
                score += mom * 500
            score = max(-100, min(100, score))
            if score >= threshold:
                matches.append((sym, score, f"score {score:+.0f}"))

        if not matches:
            return f"No tickers match the filter criteria (checked {len(symbols)} symbols)."

        matches.sort(key=lambda x: x[1], reverse=True)
        lines = [f"Filter results ({len(matches)} of {len(symbols)} tickers):"]
        for sym, val, label in matches[:15]:
            lines.append(f"  {sym:<6}  {label}")
        return "\n".join(lines)

    def _sizing(args: list[str]) -> str:
        """Show recommended position size for a ticker given current risk settings.

        Usage: /sizing SYM [price]
        Uses max_position_pct (default 2%) and ATR-14 volatility sizing.
        """
        if not args:
            return "usage: /sizing SYM [price]  (e.g. /sizing AAPL)"

        sym = args[0].upper()
        try:
            acc = broker.get_account()
            equity = float(acc.equity)
        except Exception as e:
            return f"broker error: {e!r}"

        # Get price
        if len(args) > 1:
            try:
                price = float(args[1])
            except ValueError:
                return "usage: /sizing SYM [price]"
        elif data is not None:
            try:
                bars = data.get_bars(sym, limit=2)
                price = bars[-1].close if bars else 0.0
            except Exception:
                price = 0.0
        else:
            return "Pass a price: /sizing SYM PRICE  (data client not wired)"

        if price <= 0:
            return f"Cannot size position: price is 0 for {sym}."

        max_pos_pct = 0.02
        from amms.risk.rules import position_size

        # ATR sizing if data available
        atr_val = None
        if data is not None:
            try:
                from amms.features.volatility import atr as compute_atr
                bars = data.get_bars(sym, limit=20)
                atr_val = compute_atr(bars, 14)
            except Exception:
                pass

        qty = position_size(equity, price, max_pos_pct, atr=atr_val)
        notional = qty * price
        lines = [
            f"Suggested position size for {sym}:",
            f"  Price:          ${price:.2f}",
            f"  Equity:         ${equity:,.2f}",
            f"  Max pos (2%):   ${equity * max_pos_pct:,.2f}",
            f"  Quantity:       {qty} shares",
            f"  Notional:       ${notional:,.2f}",
        ]
        if atr_val:
            lines.append(f"  ATR-14:         ${atr_val:.2f}  (volatility-adjusted)")
        else:
            lines.append("  ATR:            n/a (using max-pct only)")
        return "\n".join(lines)

    def _winloss(args: list[str]) -> str:
        """Win/loss breakdown by ticker from all filled BUY→SELL round trips.

        Usage: /winloss [SYM]
        """
        if conn is None:
            return "DB not wired."

        rows = conn.execute(
            "SELECT symbol, side, qty, filled_avg_price, filled_at "
            "FROM orders WHERE status = 'filled' AND filled_avg_price IS NOT NULL "
            "ORDER BY filled_at ASC"
        ).fetchall()

        # FIFO pairing per symbol
        buys: dict[str, list] = {}
        stats: dict[str, dict] = {}
        filter_sym = args[0].upper() if args else None

        for r in rows:
            try:
                sym = r["symbol"] if isinstance(r, __import__("sqlite3").Row) else r[0]
                side = r["side"] if isinstance(r, __import__("sqlite3").Row) else r[1]
                qty = float(r["qty"] if isinstance(r, __import__("sqlite3").Row) else r[2])
                price = float(r["filled_avg_price"] if isinstance(r, __import__("sqlite3").Row) else r[3])
            except (KeyError, TypeError, ValueError):
                try:
                    sym, side, qty, price = r[0], r[1], float(r[2]), float(r[3])
                except Exception:
                    continue

            if filter_sym and sym != filter_sym:
                continue

            if side == "buy":
                buys.setdefault(sym, []).append((price, qty))
                continue

            queue = buys.get(sym, [])
            if not queue:
                continue
            entry_price, _ = queue.pop(0)
            pnl = (price - entry_price) * qty
            s = stats.setdefault(sym, {"wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0})
            if pnl > 0:
                s["wins"] += 1
                s["gross_profit"] += pnl
            else:
                s["losses"] += 1
                s["gross_loss"] += abs(pnl)

        if not stats:
            return "No completed round trips found." + (f" (for {filter_sym})" if filter_sym else "")

        lines = ["Win/loss by ticker:"]
        all_syms = sorted(stats.keys())
        for sym in all_syms:
            s = stats[sym]
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total if total else 0
            net = s["gross_profit"] - s["gross_loss"]
            lines.append(
                f"  {sym:<6}  {s['wins']}W {s['losses']}L  "
                f"WR {wr:.0%}  net ${net:+,.2f}  "
                f"(profit ${s['gross_profit']:,.2f}  loss ${s['gross_loss']:,.2f})"
            )
        return "\n".join(lines)

    def _hold(_args: list[str]) -> str:
        """Show average holding period (in days) per ticker from filled orders.

        Pairs BUY→SELL by symbol and computes calendar days between fills.
        """
        if conn is None:
            return "DB not wired."

        rows = conn.execute(
            "SELECT symbol, side, filled_at "
            "FROM orders WHERE status = 'filled' AND filled_at IS NOT NULL "
            "ORDER BY filled_at ASC"
        ).fetchall()

        buys: dict[str, list[str]] = {}
        hold_days: dict[str, list[float]] = {}

        for r in rows:
            try:
                sym = r["symbol"]
                side = r["side"]
                filled_at = r["filled_at"]
            except (KeyError, TypeError):
                sym, side, filled_at = r[0], r[1], r[2]

            if side == "buy":
                buys.setdefault(sym, []).append(filled_at[:10])
                continue
            queue = buys.get(sym, [])
            if not queue:
                continue
            buy_date = queue.pop(0)
            try:
                from datetime import date as _date
                d1 = _date.fromisoformat(buy_date)
                d2 = _date.fromisoformat(filled_at[:10])
                days = (d2 - d1).days
                hold_days.setdefault(sym, []).append(max(0, days))
            except Exception:
                pass

        if not hold_days:
            return "No completed round trips with date data found."

        lines = ["Average holding periods:"]
        for sym in sorted(hold_days.keys()):
            days_list = hold_days[sym]
            avg = sum(days_list) / len(days_list)
            mn = min(days_list)
            mx = max(days_list)
            lines.append(
                f"  {sym:<6}  avg {avg:.1f}d  (min {mn}d, max {mx}d, {len(days_list)} trades)"
            )
        return "\n".join(lines)

    def _bb_cmd(args: list[str]) -> str:
        """Show Bollinger Bands (20,2) for open positions or a specified ticker.

        Usage: /bb [SYM]
        %B: 0 = at lower band, 0.5 = at middle, 1 = at upper band.
        >1 or <0 = price outside bands (potential reversal signal).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /bb AAPL)"

        from amms.features.bollinger import bollinger

        lines = ["Bollinger Bands (20,2):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception:
                bars = []
            bb = bollinger(bars, 20)
            if bb is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            price = bars[-1].close if bars else 0.0
            if bb.pct_b > 1.0:
                zone = "🔴 above upper band"
            elif bb.pct_b < 0.0:
                zone = "🟢 below lower band"
            elif bb.pct_b > 0.75:
                zone = "↑ near upper"
            elif bb.pct_b < 0.25:
                zone = "↓ near lower"
            else:
                zone = "↔️ mid"
            lines.append(
                f"  {sym:<6}  ${price:.2f}  "
                f"U ${bb.upper:.2f}  M ${bb.middle:.2f}  L ${bb.lower:.2f}  "
                f"%%B {bb.pct_b:.2f}  BW {bb.bandwidth:.3f}  {zone}"
            )
        return "\n".join(lines)

    def _volspike(args: list[str]) -> str:
        """Detect volume spikes for open positions or a specified ticker.

        A spike = current volume > 2× 20-day average volume.
        Usage: /volspike [SYM]
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /volspike AAPL)"

        from amms.features.bollinger import volume_spike

        lines = ["Volume spike check (vs 20-day average):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception:
                bars = []
            ratio = volume_spike(bars, 20)
            if ratio is None:
                lines.append(f"  {sym:<6}  n/a")
                continue
            if ratio >= 3.0:
                label = "🔥 extreme spike"
            elif ratio >= 2.0:
                label = "⚠️  spike"
            elif ratio >= 1.5:
                label = "↑ elevated"
            else:
                label = "→ normal"
            current_vol = bars[-1].volume if bars else 0
            lines.append(
                f"  {sym:<6}  vol {current_vol:,.0f}  ratio {ratio:.1f}×  {label}"
            )
        return "\n".join(lines)

    def _divergence(args: list[str]) -> str:
        """Detect RSI/price divergence for open positions or a ticker.

        Usage: /divergence [SYM]
        Bullish divergence: price lower low + RSI higher low → reversal up.
        Bearish divergence: price higher high + RSI lower high → reversal down.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /divergence AAPL)"

        from amms.analysis.divergence import detect_divergence

        lines = ["RSI/Price Divergence (14-period RSI, 50-bar lookback):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = detect_divergence(bars)
            if result.divergence_type == "none":
                label = "↔️  none"
            elif result.divergence_type == "bullish":
                label = "🟢 BULLISH"
            elif result.divergence_type == "bearish":
                label = "🔴 BEARISH"
            elif result.divergence_type == "hidden_bullish":
                label = "↑ hidden bullish"
            elif result.divergence_type == "hidden_bearish":
                label = "↓ hidden bearish"
            else:
                label = result.divergence_type
            conf_str = f"  conf {result.confidence:.0%}" if result.confidence > 0 else ""
            lines.append(
                f"  {sym:<6}  {label}{conf_str}  [{result.price_swing} / {result.rsi_swing}]"
            )
            if result.divergence_type != "none":
                lines.append(f"         {result.reason}")
        return "\n".join(lines)

    def _zscore_cmd(args: list[str]) -> str:
        """Show Z-score (price vs 20-bar mean/std) for positions or a ticker.

        Usage: /zscore [SYM]
        z < -2: far below mean (potential oversold)
        z > +2: far above mean (potential overbought)
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /zscore AAPL)"

        from amms.features.zscore import zscore

        lines = ["Z-score (20-bar rolling mean/std):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception:
                bars = []
            z = zscore(bars, 20)
            if z is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            price = bars[-1].close if bars else 0.0
            if z < -2.0:
                zone = "🟢 far below mean (oversold?)"
            elif z < -1.0:
                zone = "↓ below mean"
            elif z > 2.0:
                zone = "🔴 far above mean (overbought?)"
            elif z > 1.0:
                zone = "↑ above mean"
            else:
                zone = "↔️  near mean"
            lines.append(f"  {sym:<6}  ${price:.2f}  z={z:+.2f}  {zone}")
        return "\n".join(lines)

    def _vwap_cmd(args: list[str]) -> str:
        """VWAP with std dev bands for a ticker or open positions.

        Usage: /vwap [SYM]
        Shows VWAP, ±1σ/±2σ bands, deviation%, and position (above/below/at).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /vwap AAPL)"

        from amms.features.vwap import vwap_full

        lines = ["VWAP (volume-weighted avg price + σ bands):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = vwap_full(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars)")
                continue
            pos_icon = {"above_vwap": "↑", "below_vwap": "↓", "at_vwap": "↔️ "}.get(result.position, "")
            lines.append(
                f"  {sym:<6}  VWAP ${result.vwap:.2f}  "
                f"price ${result.current_price:.2f} ({result.deviation_pct:+.1f}%)  "
                f"{pos_icon} {result.position}"
            )
            lines.append(
                f"          ±1σ [{result.std1_lower:.2f}–{result.std1_upper:.2f}]  "
                f"±2σ [{result.std2_lower:.2f}–{result.std2_upper:.2f}]"
            )
        return "\n".join(lines)

    def _volprofile_cmd(args: list[str]) -> str:
        """Volume Profile: POC and Value Area for a ticker.

        Usage: /volprofile [SYM]
        POC = price level with most traded volume (70% value area).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /volprofile AAPL)"

        from amms.features.vwap import volume_profile

        lines = ["Volume Profile (POC + 70% Value Area):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = volume_profile(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars)")
                continue
            rel_icon = {"above_poc": "↑", "below_poc": "↓", "at_poc": "↔️ "}.get(result.poc_relation, "")
            lines.append(
                f"  {sym:<6}  POC ${result.poc_price:.2f}  "
                f"VA [{result.val:.2f}–{result.vah:.2f}]  "
                f"price ${result.current_price:.2f}  {rel_icon} {result.poc_relation}"
            )
        return "\n".join(lines)

    def _pairs_cmd(args: list[str]) -> str:
        """Pairs trading spread analyzer.

        Usage: /pairs SYM1 SYM2
        Computes ratio Z-score, correlation, and mean-reversion signal.
        """
        if data is None:
            return "Data client not wired."
        if len(args) < 2:
            return "usage: /pairs SYM1 SYM2  (e.g. /pairs AAPL MSFT)"

        sym1, sym2 = args[0].upper(), args[1].upper()
        try:
            bars1 = data.get_bars(sym1, limit=80)
            bars2 = data.get_bars(sym2, limit=80)
        except Exception as e:
            return f"data error: {e!r}"

        from amms.analysis.pairs_trading import analyze_pair

        result = analyze_pair(bars1, bars2)
        if result is None:
            return f"Insufficient data for {sym1}/{sym2} (need 31+ bars each)."

        signal_icon = {"long_spread": "📈", "short_spread": "📉", "neutral": "↔️ "}.get(result.signal, "")
        lines = [
            f"── Pairs: {sym1} / {sym2} ──",
            f"  Ratio:       {result.current_ratio:.4f}  (mean {result.ratio_mean:.4f}, σ {result.ratio_std:.4f})",
            f"  Z-score:     {result.ratio_zscore:+.3f}",
            f"  Correlation: {result.correlation:.3f}  (30d returns)",
            f"  Signal:      {signal_icon} {result.signal}  (strength {result.signal_strength:.2f})",
            f"  Action:      {result.recommended_action}",
        ]
        return "\n".join(lines)

    def _vregime_cmd(args: list[str]) -> str:
        """Show volatility regime classification for positions or a ticker.

        Usage: /vregime [SYM]
        Classifies ATR into low/normal/high/extreme using percentile rank vs history.
        Shows recommended position size multiplier.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /vregime AAPL)"

        from amms.analysis.volatility_regime import classify

        lines = ["Volatility Regime (ATR percentile vs history):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=100)
            except Exception:
                bars = []
            result = classify(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 30+ bars)")
                continue
            regime_icon = {"low": "🟢", "normal": "↔️ ", "high": "⚠️ ", "extreme": "🚨"}.get(result.regime, "")
            lines.append(
                f"  {sym:<6}  ATR ${result.current_atr:.2f} ({result.atr_pct_of_price:.1f}%)  "
                f"pctile {result.percentile:.0f}th  {regime_icon} {result.regime}  "
                f"→ size ×{result.recommended_size_mult:.2f}"
            )
        return "\n".join(lines)

    def _health_cmd(args: list[str]) -> str:
        """Comprehensive position health check for a single ticker.

        Usage: /health SYM
        Combines: RSI, MACD, Bollinger, ADX, Z-score, SAR, OBV, candle pattern.
        """
        if data is None:
            return "Data client not wired."

        if not args:
            # Use first open position if no arg
            try:
                positions = broker.get_positions()
                if not positions:
                    return "usage: /health SYM  (e.g. /health AAPL)"
                sym = positions[0].symbol
            except Exception as e:
                return f"broker error: {e!r}"
        else:
            sym = args[0].upper()

        try:
            bars = data.get_bars(sym, limit=80)
        except Exception as e:
            return f"data error for {sym}: {e!r}"

        if len(bars) < 5:
            return f"Insufficient data for {sym}."

        price = bars[-1].close
        lines = [f"── Health Check: {sym}  ${price:.2f} ──"]

        # RSI
        try:
            from amms.features.momentum import rsi
            rsi_val = rsi(bars, 14)
            if rsi_val is not None:
                zone = "overbought" if rsi_val > 70 else ("oversold" if rsi_val < 30 else "neutral")
                lines.append(f"  RSI-14:    {rsi_val:.1f}  ({zone})")
        except Exception:
            pass

        # Bollinger
        try:
            from amms.features.bollinger import bollinger
            bb = bollinger(bars, 20)
            if bb:
                zone = "above band" if bb.pct_b > 1 else ("below band" if bb.pct_b < 0 else f"%B {bb.pct_b:.2f}")
                lines.append(f"  Bollinger: U {bb.upper:.2f} / M {bb.middle:.2f} / L {bb.lower:.2f}  {zone}")
        except Exception:
            pass

        # MACD
        try:
            from amms.features.momentum import macd
            mc = macd(bars)
            if mc:
                ml, sl, hist = mc
                lines.append(f"  MACD:      {ml:.3f}  signal {sl:.3f}  hist {'▲' if hist > 0 else '▼'}{abs(hist):.3f}")
        except Exception:
            pass

        # ADX
        try:
            from amms.features.adx import adx as compute_adx
            adx_r = compute_adx(bars, 14)
            if adx_r:
                lines.append(f"  ADX:       {adx_r.adx:.1f} ({adx_r.trend_strength})  {adx_r.direction}")
        except Exception:
            pass

        # Z-score
        try:
            from amms.features.zscore import zscore
            z = zscore(bars, 20)
            if z is not None:
                lines.append(f"  Z-score:   {z:+.2f}")
        except Exception:
            pass

        # Parabolic SAR
        try:
            from amms.features.parabolic_sar import parabolic_sar
            sar = parabolic_sar(bars)
            if sar:
                lines.append(f"  SAR:       {sar.trend} trend  stop ${sar.sar:.2f}  ({sar.distance_pct:.1f}% away)")
        except Exception:
            pass

        # OBV
        try:
            from amms.features.obv import obv as compute_obv
            obv_r = compute_obv(bars)
            if obv_r:
                div_str = f"  {obv_r.divergence} div" if obv_r.divergence != "none" else ""
                lines.append(f"  OBV:       {obv_r.trend}{div_str}")
        except Exception:
            pass

        # Candlestick
        try:
            from amms.features.candlestick import detect_patterns
            patterns = detect_patterns(bars, lookback=3)
            if patterns:
                p = patterns[-1]
                dir_icon = "🟢" if p.direction == "bullish" else ("🔴" if p.direction == "bearish" else "↔")
                lines.append(f"  Candle:    {dir_icon} {p.name} (conf {p.confidence:.0%})")
        except Exception:
            pass

        return "\n".join(lines)

    def _candles_cmd(args: list[str]) -> str:
        """Detect candlestick patterns for open positions or a ticker.

        Usage: /candles [SYM]
        Detects: Doji, Hammer, Shooting Star, Marubozu, Spinning Top,
        Bullish/Bearish Engulfing, Harami.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /candles AAPL)"

        from amms.features.candlestick import detect_patterns

        lines = ["Candlestick Pattern Analysis (last 5 bars):"]
        found_any = False
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=10)
            except Exception:
                bars = []
            patterns = detect_patterns(bars, lookback=5)
            price = bars[-1].close if bars else 0.0
            if not patterns:
                lines.append(f"  {sym:<6}  ${price:.2f}  no significant patterns")
            else:
                found_any = True
                for p in patterns:
                    dir_icon = "🟢" if p.direction == "bullish" else ("🔴" if p.direction == "bearish" else "↔️ ")
                    lines.append(
                        f"  {sym:<6}  ${price:.2f}  {dir_icon} {p.name}  "
                        f"(conf {p.confidence:.0%})"
                    )
                    lines.append(f"         {p.description}")
        return "\n".join(lines)

    def _fib_cmd(args: list[str]) -> str:
        """Show Fibonacci retracement and extension levels for a ticker.

        Usage: /fib [SYM]
        Auto-detects swing high/low from last 50 bars.
        Shows key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) with nearest
        support and resistance highlighted.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /fib AAPL)"

        from amms.features.fibonacci import fibonacci

        lines = []
        for sym in symbols[:3]:
            try:
                bars = data.get_bars(sym, limit=55)
            except Exception:
                bars = []
            result = fibonacci(bars)
            if result is None:
                lines.append(f"{sym}: n/a (not enough history)")
                continue

            dir_label = "▲ uptrend" if result.direction == "uptrend" else "▼ downtrend"
            lines.append(
                f"── {sym} Fibonacci ({dir_label}  "
                f"swing L ${result.swing_low:.2f}  H ${result.swing_high:.2f}) ──"
            )
            lines.append(f"  Current price: ${result.current_price:.2f}")
            lines.append("")

            for lv in result.levels:
                marker = ""
                if result.nearest_support and abs(lv.price - result.nearest_support.price) < 0.001:
                    marker = "  ← support"
                elif result.nearest_resistance and abs(lv.price - result.nearest_resistance.price) < 0.001:
                    marker = "  ← resistance"
                lines.append(f"  {lv.label:>10}  ${lv.price:.2f}{marker}")

        return "\n".join(lines)

    def _journalstats_cmd(_args: list[str]) -> str:
        """Extended trade journal statistics.

        Shows win rate, expectancy, profit factor, Sharpe, streaks, and more
        from all completed trade pairs.
        """
        if conn is None:
            return "DB not wired."

        from amms.analysis.journal_stats import compute

        stats = compute(conn)
        if stats is None:
            return "No completed trades found."

        pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor != float("inf") else "∞"
        sharpe_str = f"{stats.sharpe:.2f}" if stats.sharpe is not None else "n/a"
        hold_str = f"{stats.avg_hold_days:.1f}d" if stats.avg_hold_days is not None else "n/a"

        lines = [
            f"Trade Journal Statistics ({stats.n_trades} trades):",
            f"  Total P&L:       ${stats.total_pnl:+,.2f}",
            f"  Win rate:        {stats.win_rate:.1%}",
            f"  Avg win:         ${stats.avg_win:+,.2f}",
            f"  Avg loss:        -${stats.avg_loss:,.2f}",
            f"  Profit factor:   {pf_str}",
            f"  Expectancy:      ${stats.expectancy:+,.2f} / trade",
            f"  Sharpe:          {sharpe_str}",
            f"  Largest win:     ${stats.largest_win:,.2f}",
            f"  Largest loss:    -${stats.largest_loss:,.2f}",
            f"  Max win streak:  {stats.max_win_streak}",
            f"  Max loss streak: {stats.max_loss_streak}",
            f"  Avg hold time:   {hold_str}",
        ]

        if stats.profit_factor > 2.0:
            lines.append("  ✅ Excellent profit factor (>2.0)")
        elif stats.profit_factor > 1.5:
            lines.append("  ↔️  Good profit factor (1.5–2.0)")
        elif stats.profit_factor < 1.0:
            lines.append("  🔴 Profit factor < 1.0 — strategy losing money")

        return "\n".join(lines)

    def _stress_cmd(args: list[str]) -> str:
        """Portfolio stress test: how would current positions do in a crash?

        Usage: /stress [SCENARIO]
        Scenarios: 2008_crisis, 2020_covid, dotcom_crash, flash_crash_2010,
                   rising_rates, mild_correction, severe_bear, custom:-20
        Default: 2008_crisis
        """
        from amms.analysis.stress_test import SCENARIOS, stress_test

        scenario = "2008_crisis"
        custom_shock = None

        if args:
            arg = args[0].lower()
            if arg.startswith("custom:"):
                try:
                    pct = float(arg.split(":")[1])
                    custom_shock = pct / 100
                    scenario = "custom"
                except (IndexError, ValueError):
                    return "Custom shock format: custom:-20 (for -20%)"
            elif arg in SCENARIOS:
                scenario = arg
            else:
                valid = ", ".join(SCENARIOS.keys())
                return f"Unknown scenario. Valid: {valid}, custom:-N"

        result = stress_test(broker, scenario, custom_shock_pct=custom_shock)
        if result is None:
            return "Could not run stress test (no positions or broker error)."

        if result.verdict == "critical":
            icon = "🚨"
        elif result.verdict == "severe":
            icon = "🔴"
        elif result.verdict == "moderate":
            icon = "⚠️ "
        else:
            icon = "✅"

        lines = [
            f"{icon} Stress Test: {result.scenario.replace('_', ' ').title()}",
            f"   Shock: {result.shock_pct:+.0f}%",
            f"   Portfolio MV now: ${result.initial_total_mv:,.2f}",
            f"   Portfolio MV after: ${result.stressed_total_mv:,.2f}",
            f"   Estimated loss: ${result.total_loss:,.2f} ({result.total_loss_pct:+.1f}%)",
            f"   Risk level: {result.verdict.replace('_', ' ')}",
            "",
            "   Per position:",
        ]
        for pos in sorted(result.positions, key=lambda x: x.loss):
            lines.append(
                f"     {pos.symbol:<6}  ${pos.current_value:>10,.2f} → "
                f"${pos.stressed_value:>10,.2f}  ({pos.loss_pct:+.1f}%  loss ${abs(pos.loss):,.2f})"
            )

        return "\n".join(lines)

    def _strategy_selector_cmd(_args: list[str]) -> str:
        """Recommend the best strategy for current market conditions.

        Uses market regime, ADX, SPY RSI, and sector rotation to suggest
        which registered strategy to run.
        """
        if data is None:
            return "Data client not wired."

        from amms.analysis.strategy_selector import recommend

        # Gather inputs
        regime = "unknown"
        vixy = None
        spy_rsi = None
        adx_val = None
        top_sector = None

        try:
            from amms.analysis.regime import detect_regime
            r = detect_regime(data)
            regime = r.label
            vixy = r.vixy_1d_pct
        except Exception:
            pass

        try:
            from amms.features.momentum import rsi
            spy_bars = data.get_bars("SPY", limit=20)
            spy_rsi = rsi(spy_bars, 14)
        except Exception:
            pass

        try:
            from amms.features.adx import adx
            spy_bars = data.get_bars("SPY", limit=50)
            adx_r = adx(spy_bars, 14)
            if adx_r:
                adx_val = adx_r.adx
        except Exception:
            pass

        try:
            from amms.analysis.sector_rotation import detect_rotation
            sectors = detect_rotation(data, n=20)
            in_sectors = [s.etf for s in sectors if s.trend == "in"]
            top_sector = in_sectors[0] if in_sectors else None
        except Exception:
            pass

        rec = recommend(regime, vix_proxy=vixy, adx=adx_val, spy_rsi=spy_rsi, top_sector=top_sector)

        lines = [
            "Strategy Recommendation:",
            f"  Regime: {rec.regime}  (risk ×{rec.risk_multiplier:.2f})",
            f"  ✅ Primary:   {rec.primary}",
        ]
        if rec.secondary:
            lines.append(f"  ↔️  Secondary: {rec.secondary}")
        if rec.avoid:
            lines.append(f"  ❌ Avoid:     {', '.join(rec.avoid)}")
        lines.append("")
        lines.append("  Reasoning:")
        for r_line in rec.reasoning:
            lines.append(f"    • {r_line}")

        return "\n".join(lines)

    def _scan2_cmd(args: list[str]) -> str:
        """Multi-indicator signal scanner for the watchlist.

        Usage: /scan2 [min_score]
        Scans all watchlist symbols for BB + RSI + Stoch + MACD + Z-score setups.
        min_score: minimum combined signal score (default 3.0, max 10)
        """
        if data is None:
            return "Data client not wired."

        min_score = 3.0
        if args:
            try:
                min_score = float(args[0])
            except ValueError:
                pass

        # Get watchlist symbols
        from amms.data.dynamic_watchlist import load as load_watchlist
        watchlist = list(load_watchlist())
        if not watchlist:
            return "Watchlist is empty — use /setlist to add symbols."

        from amms.analysis.signal_scanner import scan_signals

        setups = scan_signals(watchlist, data, min_score=min_score, top_n=10)

        if not setups:
            return f"No setups found with score >= {min_score:.0f}. Try lowering the threshold."

        buys = [s for s in setups if s.direction == "buy"]
        sells = [s for s in setups if s.direction == "sell"]

        lines = [f"Signal Scanner (watchlist: {len(watchlist)} symbols  min score: {min_score:.0f}):"]

        if buys:
            lines.append("  🟢 Buy Setups:")
            for s in buys[:5]:
                lines.append(
                    f"    {s.symbol:<6}  ${s.price:.2f}  score {s.score:.1f}/10  [{s.confidence}]"
                )
                for sig in s.signals[:3]:
                    lines.append(f"      • {sig}")

        if sells:
            lines.append("  🔴 Sell/Short Setups:")
            for s in sells[:5]:
                lines.append(
                    f"    {s.symbol:<6}  ${s.price:.2f}  score {s.score:.1f}/10  [{s.confidence}]"
                )
                for sig in s.signals[:3]:
                    lines.append(f"      • {sig}")

        if not buys and not sells:
            lines.append("  No directional setups above threshold.")

        return "\n".join(lines)

    def _montecarlo_cmd(args: list[str]) -> str:
        """Monte Carlo simulation from trade history.

        Usage: /montecarlo [N_TRADES]
        Samples from realized trade P&L history, runs 1000 simulations,
        shows probability of ruin, expected drawdown, and equity range.
        """
        if conn is None:
            return "DB not wired."

        # Load realized trade returns from journal
        try:
            rows = conn.execute(
                "SELECT buy_price, sell_price, qty FROM trade_pairs ORDER BY sell_ts DESC LIMIT 200"
            ).fetchall()
        except Exception:
            rows = []

        if len(rows) < 5:
            return f"Need at least 5 completed trades for simulation (found {len(rows)})."

        trade_returns = []
        for row in rows:
            try:
                buy_p = float(row[0])
                sell_p = float(row[1])
                if buy_p > 0:
                    trade_returns.append((sell_p - buy_p) / buy_p)
            except (TypeError, ValueError, IndexError):
                continue

        if len(trade_returns) < 5:
            return "Not enough valid trade data for simulation."

        n_trades_arg = None
        if args:
            try:
                n_trades_arg = int(args[0])
            except ValueError:
                pass

        try:
            equity = float(broker.get_account().equity)
        except Exception:
            equity = 100_000.0

        from amms.analysis.monte_carlo import simulate

        result = simulate(
            trade_returns,
            n_simulations=1000,
            n_trades=n_trades_arg,
            initial_equity=equity,
        )
        if result is None:
            return "Simulation failed."

        lines = [
            f"Monte Carlo Simulation ({result.n_simulations}× {result.n_trades} trades):",
            f"  Starting equity: ${result.initial_equity:,.2f}",
            "",
            "  Final equity range (1000 paths):",
            f"    5th percentile:  ${result.p5_final:,.2f}  ({(result.p5_final / result.initial_equity - 1) * 100:+.1f}%)",
            f"    Median (50th):   ${result.p50_final:,.2f}  ({(result.p50_final / result.initial_equity - 1) * 100:+.1f}%)",
            f"    95th percentile: ${result.p95_final:,.2f}  ({(result.p95_final / result.initial_equity - 1) * 100:+.1f}%)",
            "",
            "  Drawdown risk:",
            f"    Median max DD:  {result.median_max_drawdown_pct:.1f}%",
            f"    95th pct max DD: {result.p95_max_drawdown_pct:.1f}%",
            f"    P(DD > 20%):    {result.prob_20pct_dd:.1%}",
            f"    P(ruin <50%):   {result.prob_ruin:.1%}",
            f"    P(profitable):  {result.prob_positive:.1%}",
        ]

        if result.prob_ruin > 0.1:
            lines.append("  🚨 High ruin probability — review position sizing!")
        elif result.prob_20pct_dd > 0.3:
            lines.append("  ⚠️  Significant drawdown risk — consider tighter stops.")
        elif result.prob_positive > 0.7:
            lines.append("  ✅ Strategy shows positive expectancy.")

        return "\n".join(lines)

    def _kelly_cmd(args: list[str]) -> str:
        """Kelly criterion position sizing.

        Usage: /kelly PRICE STOP_PCT [WIN_RATE] [AVG_WIN_PCT] [AVG_LOSS_PCT]
        Example: /kelly 150 2.0 0.55 3.0 1.5
          PRICE: current price
          STOP_PCT: stop loss distance as % of price
          WIN_RATE: historical win rate (default 0.5)
          AVG_WIN_PCT: avg winning trade return % (default 3.0)
          AVG_LOSS_PCT: avg losing trade loss % (default 1.5)
        """
        if len(args) < 2:
            return "usage: /kelly PRICE STOP_PCT [WIN_RATE] [AVG_WIN_PCT] [AVG_LOSS_PCT]\nexample: /kelly 150 2.0 0.55 3.0 1.5"

        try:
            price = float(args[0])
            stop_pct = float(args[1])
            win_rate = float(args[2]) if len(args) > 2 else 0.50
            avg_win = float(args[3]) if len(args) > 3 else 3.0
            avg_loss = float(args[4]) if len(args) > 4 else 1.5
        except ValueError:
            return "Invalid number — usage: /kelly PRICE STOP_PCT [WIN_RATE] [AVG_WIN_PCT] [AVG_LOSS_PCT]"

        try:
            acc = broker.get_account()
            equity = float(acc.equity)
        except Exception as e:
            return f"broker error: {e!r}"

        from amms.risk.position_sizing import fixed_fraction, kelly_criterion

        kelly = kelly_criterion(equity, price, stop_pct, win_rate=win_rate, avg_win_pct=avg_win, avg_loss_pct=avg_loss)
        ff = fixed_fraction(equity, price, stop_pct, risk_pct=1.0)

        lines = [
            f"Position Sizing Comparison (equity ${equity:,.2f}):",
            f"  Price: ${price:.2f}  Stop: {stop_pct:.1f}%",
            f"  Win rate: {win_rate:.0%}  Avg win: {avg_win:.1f}%  Avg loss: {avg_loss:.1f}%",
            "",
            f"  Kelly (¼ Kelly):  {kelly.shares} shares  ${kelly.dollar_amount:,.2f}  "
            f"({kelly.pct_of_equity:.1f}% equity)  risk ${kelly.risk_amount:.2f}",
            f"  Fixed 1% risk:    {ff.shares} shares  ${ff.dollar_amount:,.2f}  "
            f"({ff.pct_of_equity:.1f}% equity)  risk ${ff.risk_amount:.2f}",
            f"  {kelly.notes}",
        ]
        if kelly.shares == 0:
            lines.append("  ⚠️  Kelly says no edge — skip this trade.")
        return "\n".join(lines)

    def _rr_calc(args: list[str]) -> str:
        """Calculate risk/reward ratio for a trade.

        Usage: /rr ENTRY STOP TARGET [QTY]
        Example: /rr 150.00 145.00 162.00 100
          ENTRY: entry price
          STOP: stop loss price
          TARGET: profit target price
          QTY: optional share count (default: use /sizing logic)
        """
        if len(args) < 3:
            return "usage: /rr ENTRY STOP TARGET [QTY]\nexample: /rr 150 145 162 100"

        try:
            entry = float(args[0])
            stop = float(args[1])
            target = float(args[2])
            qty = int(args[3]) if len(args) > 3 else None
        except ValueError:
            return "Invalid number — usage: /rr ENTRY STOP TARGET [QTY]"

        if entry <= 0 or stop >= entry:
            return "STOP must be below ENTRY for a long trade."
        if target <= entry:
            return "TARGET must be above ENTRY for a long trade."

        risk_per_share = entry - stop
        reward_per_share = target - entry
        rr_ratio = reward_per_share / risk_per_share

        risk_pct = risk_per_share / entry * 100
        reward_pct = reward_per_share / entry * 100

        lines = [
            f"Risk/Reward Analysis:",
            f"  Entry:  ${entry:.2f}",
            f"  Stop:   ${stop:.2f}  (−${risk_per_share:.2f} / −{risk_pct:.1f}%)",
            f"  Target: ${target:.2f}  (+${reward_per_share:.2f} / +{reward_pct:.1f}%)",
            f"  R:R ratio: 1 : {rr_ratio:.2f}",
        ]

        if rr_ratio < 1.5:
            lines.append("  ⚠️  Poor R:R (<1.5) — consider skipping.")
        elif rr_ratio < 2.0:
            lines.append("  ↔️  Acceptable R:R (1.5–2.0).")
        elif rr_ratio < 3.0:
            lines.append("  ✅ Good R:R (2.0–3.0).")
        else:
            lines.append("  🎯 Excellent R:R (>3.0).")

        if qty:
            risk_total = qty * risk_per_share
            reward_total = qty * reward_per_share
            lines.append(f"  {qty} shares → max risk ${risk_total:,.2f} / max reward ${reward_total:,.2f}")

            # Break-even win rate needed
            be_win_rate = 1 / (1 + rr_ratio)
            lines.append(f"  Break-even win rate: {be_win_rate:.1%}")

        return "\n".join(lines)

    def _obv_cmd(args: list[str]) -> str:
        """Show On-Balance Volume trend for positions or a ticker.

        Usage: /obv [SYM]
        Rising OBV = buying pressure. Falling = selling.
        Divergence = OBV and price moving in opposite directions.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /obv AAPL)"

        from amms.features.obv import obv as compute_obv

        lines = ["On-Balance Volume (OBV):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=30)
            except Exception:
                bars = []
            result = compute_obv(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a")
                continue
            trend_label = "▲ rising" if result.trend == "rising" else ("▼ falling" if result.trend == "falling" else "↔ flat")
            div_label = ""
            if result.divergence == "bullish":
                div_label = "  🟢 bullish div (accumulation)"
            elif result.divergence == "bearish":
                div_label = "  🔴 bearish div (distribution)"
            lines.append(
                f"  {sym:<6}  OBV {result.obv:+,.0f}  EMA {result.obv_ema:+,.0f}  {trend_label}{div_label}"
            )
        return "\n".join(lines)

    def _attribution_cmd(_args: list[str]) -> str:
        """Show performance attribution: which positions drive portfolio returns.

        Shows each position's weight, P&L, and contribution to total return.
        """
        try:
            from amms.analysis.performance_attribution import compute
            report = compute(broker)
        except Exception as e:
            return f"attribution error: {e!r}"

        if not report.rows:
            return "no open positions"

        lines = [
            f"Performance Attribution (total P&L: {report.total_unrealized_pnl:+.2f}  "
            f"portfolio return: {report.total_return_pct:+.2f}%):"
        ]
        for r in report.rows:
            bar_len = int(abs(r.contribution_pct) * 20)
            bar = ("█" * bar_len).ljust(4)
            sign = "+" if r.contribution_pct >= 0 else ""
            lines.append(
                f"  {r.symbol:<6}  {r.weight_pct:5.1f}% wt  "
                f"P&L {r.unrealized_pnl:+8.2f} ({r.unrealized_pnl_pct:+.1f}%)  "
                f"contrib {sign}{r.contribution_pct:.2f}%  [{bar}]"
            )
        if report.top_contributor:
            lines.append(f"Best: {report.top_contributor}")
        if report.top_detractor:
            lines.append(f"Worst: {report.top_detractor}")
        return "\n".join(lines)

    def _sar_cmd(args: list[str]) -> str:
        """Show Parabolic SAR for open positions or a ticker.

        Usage: /sar [SYM]
        SAR below price = uptrend. SAR above price = downtrend.
        Distance % shows how far price is from the stop level.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /sar AAPL)"

        from amms.features.parabolic_sar import parabolic_sar

        lines = ["Parabolic SAR (AF=0.02, step=0.02, max=0.20):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                bars = []
            result = parabolic_sar(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a")
                continue
            price = bars[-1].close if bars else 0.0
            if result.trend == "up":
                label = f"▲ uptrend  SAR ${result.sar:.2f} below price ({result.distance_pct:.1f}% buffer)"
            else:
                label = f"▼ downtrend  SAR ${result.sar:.2f} above price ({result.distance_pct:.1f}% below)"
            lines.append(f"  {sym:<6}  ${price:.2f}  {label}  AF {result.acceleration:.3f}")
        return "\n".join(lines)

    def _trend_cmd(args: list[str]) -> str:
        """Multi-indicator trend summary for a position or ticker.

        Usage: /trend [SYM]
        Shows SMA-50/200, EMA-20, RSI-14, MACD, ADX in one compact view.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions[:3]]  # limit to 3 for readability
        else:
            return "no open positions (pass a ticker: /trend AAPL)"

        from amms.features.momentum import ema, macd, rsi, sma

        lines = []
        for sym in symbols[:3]:
            try:
                bars = data.get_bars(sym, limit=210)
            except Exception:
                bars = []

            price = bars[-1].close if bars else 0.0
            lines.append(f"── {sym} ${price:.2f} ──")

            # SMA 50/200
            sma50 = sma(bars, 50)
            sma200 = sma(bars, 200)
            if sma50 and sma200:
                cross = "▲ golden cross" if sma50 > sma200 else "▼ death cross"
                lines.append(f"  SMA50 {sma50:.2f}  SMA200 {sma200:.2f}  {cross}")
            elif sma50:
                above = "above" if price > sma50 else "below"
                lines.append(f"  SMA50 {sma50:.2f}  price {above} SMA50")

            # EMA 20
            ema20 = ema(bars, 20)
            if ema20:
                dir_e = "▲" if price > ema20 else "▼"
                lines.append(f"  EMA20 {ema20:.2f}  price {dir_e} EMA20")

            # RSI
            rsi_val = rsi(bars, 14)
            if rsi_val is not None:
                rsi_zone = "overbought" if rsi_val > 70 else ("oversold" if rsi_val < 30 else "neutral")
                lines.append(f"  RSI14 {rsi_val:.1f}  ({rsi_zone})")

            # MACD
            macd_result = macd(bars)
            if macd_result:
                ml, sl, hist = macd_result
                hist_dir = "▲" if hist > 0 else "▼"
                lines.append(f"  MACD {ml:.3f}  signal {sl:.3f}  hist {hist_dir}{abs(hist):.3f}")

            # ADX
            try:
                from amms.features.adx import adx as compute_adx
                adx_result = compute_adx(bars, 14)
                if adx_result:
                    lines.append(f"  ADX {adx_result.adx:.1f} ({adx_result.trend_strength})  {adx_result.direction}")
            except Exception:
                pass

        return "\n".join(lines)

    def _ichimoku_cmd(args: list[str]) -> str:
        """Show Ichimoku Cloud for open positions or a ticker.

        Usage: /ichimoku [SYM]
        Shows Tenkan/Kijun lines, cloud position, and momentum signal.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /ichimoku AAPL)"

        from amms.features.ichimoku import ichimoku

        lines = ["Ichimoku Cloud (9/26/52):"]
        for sym in symbols[:6]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = ichimoku(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 52+ bars)")
                continue
            if result.position == "above_cloud":
                pos_label = "🟢 above cloud"
            elif result.position == "below_cloud":
                pos_label = "🔴 below cloud"
            else:
                pos_label = "🟡 in cloud"
            cloud_label = "☁️ green" if result.cloud_color == "green" else ("☁️ red" if result.cloud_color == "red" else "☁️ flat")
            mom_label = "▲ T>K" if result.momentum == "bullish" else ("▼ T<K" if result.momentum == "bearish" else "↔ T=K")
            lines.append(
                f"  {sym:<6}  ${result.price:.2f}  T {result.tenkan:.2f}  K {result.kijun:.2f}  "
                f"SpA {result.span_a:.2f}  SpB {result.span_b:.2f}  "
                f"{pos_label}  {cloud_label}  {mom_label}"
            )
        return "\n".join(lines)

    def _watchdog_cmd(_args: list[str]) -> str:
        """Run the daily risk watchdog report.

        Checks: circuit breaker, regime, open P&L, Bollinger/RSI extremes,
        ADX strong trends, sector rotation.
        """
        from amms.analysis.watchdog import generate

        try:
            report = generate(broker, conn=conn, data=data)
        except Exception as e:
            return f"watchdog error: {e!r}"

        lines = [f"🐕 Watchdog Report — {report.summary}", ""]

        if report.circuit_open:
            lines.append(f"🚨 Circuit Breaker: OPEN ({report.circuit_reason})")

        lines.append(f"📊 Regime: {report.regime}  (risk ×{report.regime_risk_multiplier:.2f})")
        lines.append(f"💼 Positions: {report.position_count}  Open P&L: {report.total_open_pnl:+.2f} ({report.total_open_pnl_pct:+.1f}%)")

        if report.top_rotating_in:
            lines.append(f"🔄 Rotating in: {', '.join(report.top_rotating_in)}")

        if report.warnings:
            lines.append("")
            lines.append("⚠️  Warnings:")
            for w in report.warnings[:10]:
                icon = "🚨" if w.level == "critical" else ("⚠️ " if w.level == "warning" else "ℹ️ ")
                lines.append(f"  {icon} [{w.symbol}] {w.message}")
        else:
            lines.append("✅ No warnings.")

        return "\n".join(lines)

    def _stoch_cmd(args: list[str]) -> str:
        """Show Stochastic %K/%D for open positions or a ticker.

        Usage: /stoch [SYM]
        %K < 20: oversold.  %K > 80: overbought.
        Bullish cross: %K crosses %D upward while oversold.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /stoch AAPL)"

        from amms.features.stochastic import stochastic

        lines = ["Stochastic Oscillator (14,3):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception:
                bars = []
            result = stochastic(bars, 14, 3)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            if result.zone == "oversold":
                zone_label = "🟢 oversold"
            elif result.zone == "overbought":
                zone_label = "🔴 overbought"
            else:
                zone_label = "↔️  neutral"
            sig_label = ""
            if result.signal == "bullish_cross":
                sig_label = "  ⬆️  bullish cross!"
            elif result.signal == "bearish_cross":
                sig_label = "  ⬇️  bearish cross!"
            lines.append(
                f"  {sym:<6}  %%K {result.k:.1f}  %%D {result.d:.1f}  {zone_label}{sig_label}"
            )
        return "\n".join(lines)

    def _confluence_cmd(args: list[str]) -> str:
        """Show multi-indicator confluence score for positions or a ticker.

        Usage: /confluence [SYM]
        Aggregates RSI, MACD, Bollinger, ADX, Stochastic, Z-score, momentum
        into a single bullish/bearish score (-1 to +1).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /confluence AAPL)"

        from amms.analysis.confluence import analyze

        lines = ["Signal Confluence (multi-indicator score):"]
        for sym in symbols[:6]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            if len(bars) < 5:
                lines.append(f"  {sym:<6}  n/a")
                continue
            result = analyze(bars)
            if result.verdict == "strong_buy":
                vdict = "🟢🟢 STRONG BUY"
            elif result.verdict == "buy":
                vdict = "🟢 buy"
            elif result.verdict == "strong_sell":
                vdict = "🔴🔴 STRONG SELL"
            elif result.verdict == "sell":
                vdict = "🔴 sell"
            else:
                vdict = "↔️  neutral"
            score_bar = "+" * max(0, int(result.score * 5)) if result.score >= 0 else "-" * max(0, int(-result.score * 5))
            lines.append(
                f"  {sym:<6}  score {result.score:+.2f}  [{score_bar:<5}]  {vdict}  "
                f"({result.signals_checked} signals  conf {result.confidence:.0%})"
            )
            for s in result.bullish_signals[:3]:
                lines.append(f"    ✅ {s}")
            for s in result.bearish_signals[:3]:
                lines.append(f"    ❌ {s}")
        return "\n".join(lines)

    def _adx_cmd(args: list[str]) -> str:
        """Show ADX (Average Directional Index) trend strength for positions or a ticker.

        Usage: /adx [SYM]
        ADX < 20: choppy/ranging.  ADX 20-40: trending.  ADX > 40: strong trend.
        +DI > -DI: bullish direction.  -DI > +DI: bearish direction.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /adx AAPL)"

        from amms.features.adx import adx as compute_adx

        lines = ["ADX Trend Strength (14-period):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                bars = []
            result = compute_adx(bars, 14)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (not enough history)")
                continue
            if result.trend_strength == "none":
                strength_label = "↔️  ranging"
            elif result.trend_strength == "emerging":
                strength_label = "↗️  emerging trend"
            elif result.trend_strength == "strong":
                strength_label = "📈 strong trend"
            elif result.trend_strength == "very_strong":
                strength_label = "🔥 very strong trend"
            else:
                strength_label = "⚡ extreme trend"
            dir_label = "▲ bull" if result.direction == "bullish" else ("▼ bear" if result.direction == "bearish" else "↔ neutral")
            lines.append(
                f"  {sym:<6}  ADX {result.adx:.1f}  +DI {result.plus_di:.1f}  -DI {result.minus_di:.1f}"
                f"  {strength_label}  {dir_label}"
            )
        return "\n".join(lines)

    def _rotation(_args: list[str]) -> str:
        """Show sector rotation: which SPDR sector ETFs are outperforming SPY.

        Uses 20-day momentum relative to SPY as benchmark.
        """
        if data is None:
            return "Data client not wired."
        try:
            from amms.analysis.sector_rotation import detect_rotation
            sectors = detect_rotation(data, n=20)
        except Exception as e:
            return f"sector rotation error: {e!r}"

        if not sectors:
            return "No sector rotation data available."

        lines = ["Sector rotation (20d momentum vs SPY):"]
        for s in sectors:
            if s.momentum_20d is None:
                lines.append(f"  {s.sector:<22} {s.etf:<5}  n/a")
                continue
            trend_sym = "📈 in" if s.trend == "in" else ("📉 out" if s.trend == "out" else "↔️ neutral")
            vs_str = f"{s.vs_spy:+.1f}% vs SPY" if s.vs_spy is not None else ""
            lines.append(
                f"  {s.sector:<22} {s.etf:<5}  {s.momentum_20d:+.1f}%  {vs_str}  {trend_sym}"
            )
        return "\n".join(lines)

    def _risk2r(_args: list[str]) -> str:
        """Show current open P&L for each position in terms of risk units (R).

        R = amount risked per trade (stop-loss distance × qty).
        +1R means the trade has made back its initial risk. +2R = 2× risk as profit.
        """
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"
        if not positions:
            return "no open positions"

        stop_pct = 0.03  # default
        if conn is not None:
            try:
                from amms.runtime_overrides import get_overrides
                overrides = get_overrides(conn)
                sl = overrides.get("stop_loss", 0.0)
                if sl and float(sl) > 0:
                    stop_pct = float(sl)
            except Exception:
                pass

        lines = [f"Open P&L in R-units (stop {stop_pct:.1%}):"]
        for p in positions:
            try:
                entry = float(p.avg_entry_price)
                unreal_pl = float(p.unrealized_pl)
                qty = float(p.qty)
                if qty <= 0 or entry <= 0:
                    continue
                risk_per_share = entry * stop_pct
                risk_total = risk_per_share * qty
                r_value = unreal_pl / risk_total if risk_total > 0 else 0.0
                sign = "+" if r_value >= 0 else ""
                emoji = "📈" if r_value >= 2 else ("✅" if r_value >= 1 else ("⚠️" if r_value < 0 else "↔️"))
                lines.append(
                    f"  {p.symbol:<6}  {sign}{r_value:.2f}R  "
                    f"(P&L ${unreal_pl:+,.2f}  risk ${risk_total:,.2f})  {emoji}"
                )
            except (TypeError, ValueError, ZeroDivisionError):
                lines.append(f"  {p.symbol}: data error")
        return "\n".join(lines)

    def _stopopt(args: list[str]) -> str:
        """Suggest optimal stop-loss % for a ticker based on historical ATR.

        Usage: /stopopt [SYM]  — held positions or specific ticker
        Shows tight (1×ATR), balanced (1.5×ATR), and wide (2×ATR) stops.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if args:
            symbols = [args[0].upper()]
        elif positions:
            symbols = [p.symbol for p in positions]
        else:
            return "no open positions (pass a ticker: /stopopt AAPL)"

        from amms.analysis.stop_optimizer import suggest_stop

        lines = ["Stop-loss suggestions (ATR-14 based):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception:
                bars = []
            s = suggest_stop(sym, bars)
            if s.atr_pct is None:
                lines.append(f"  {sym:<6}  n/a (not enough data)")
                continue
            lines.append(
                f"  {sym:<6}  price ${s.current_price:.2f}  "
                f"ATR {s.atr_pct:.2f}%  "
                f"tight {s.stop_tight_pct:.1f}%  "
                f"balanced {s.stop_balanced_pct:.1f}%  "
                f"wide {s.stop_wide_pct:.1f}%"
            )
            lines.append(f"         → {s.recommendation}")
        return "\n".join(lines)

    def _chart(_args: list[str]) -> str:
        """ASCII equity curve chart from equity_snapshots (last 30 days).

        Uses 8 block characters to represent the equity curve as a sparkline.
        """
        if conn is None:
            return "DB not wired."
        try:
            rows = conn.execute(
                "SELECT substr(ts,1,10) AS day, equity "
                "FROM equity_snapshots "
                "ORDER BY ts ASC"
            ).fetchall()
        except Exception as e:
            return f"DB error: {e!r}"

        equities = []
        dates = []
        for r in rows:
            try:
                dates.append(r["day"])
                equities.append(float(r["equity"]))
            except (KeyError, TypeError):
                try:
                    dates.append(r[0])
                    equities.append(float(r[1]))
                except Exception:
                    pass

        if len(equities) < 2:
            return "Not enough equity history for chart."

        equities = equities[-30:]
        dates = dates[-30:]

        min_e = min(equities)
        max_e = max(equities)
        rang = max_e - min_e or 1.0
        blocks = "▁▂▃▄▅▆▇█"

        chart_line = ""
        for e in equities:
            idx = int((e - min_e) / rang * (len(blocks) - 1))
            chart_line += blocks[idx]

        start_e = equities[0]
        end_e = equities[-1]
        ret = (end_e - start_e) / start_e * 100

        lines = [
            f"Equity curve ({len(equities)} snapshots):",
            f"  {chart_line}",
            f"  {dates[0]} → {dates[-1]}",
            f"  ${start_e:,.0f} → ${end_e:,.0f}  ({ret:+.1f}%)",
        ]
        return "\n".join(lines)

    def _strategies(_args: list[str]) -> str:
        """List all registered trading strategies with their default parameters."""
        from amms.strategy.base import registered_strategies
        registry = registered_strategies()
        if not registry:
            return "No strategies registered."
        lines = [f"Registered strategies ({len(registry)}):"]
        for name in sorted(registry.keys()):
            cls = registry[name]
            try:
                instance = cls()
                params = {
                    k: v for k, v in vars(instance).items()
                    if k != "name" and not k.startswith("_")
                }
                param_str = "  ".join(f"{k}={v}" for k, v in params.items())
            except Exception:
                param_str = "(custom init)"
            lines.append(f"  {name:<20}  {param_str or '(no params)'}")
        return "\n".join(lines)

    def _forecast_cmd(args: list[str]) -> str:
        """Statistical price forecast with confidence intervals.

        Usage: /forecast [SYM] [DAYS]
        Uses historical volatility to project 68%% and 95%% price ranges.
        Default: 10 trading days. NOT a prediction — statistical baseline only.
        """
        if data is None:
            return "Data client not wired."

        sym = None
        days = 10
        if args:
            for arg in args:
                if arg.isdigit():
                    days = int(arg)
                else:
                    sym = arg.upper()

        if sym is None:
            try:
                positions = broker.get_positions()
                if positions:
                    sym = positions[0].symbol
            except Exception:
                pass

        if sym is None:
            return "usage: /forecast [SYM] [DAYS]  (e.g. /forecast AAPL 10)"

        days = max(1, min(days, 90))
        try:
            bars = data.get_bars(sym, limit=40)
        except Exception as e:
            return f"data error for {sym}: {e!r}"

        from amms.analysis.price_forecast import forecast as pf_forecast

        result = pf_forecast(bars, horizon_days=days)
        if result is None:
            return f"Insufficient data for {sym} (need 31+ bars)."

        change_pct = (result.expected - result.current_price) / result.current_price * 100

        lines = [
            f"── Price Forecast: {sym}  ({days}d horizon) ──",
            f"  Current:    ${result.current_price:.2f}",
            f"  Expected:   ${result.expected:.2f}  ({change_pct:+.1f}%%)",
            f"  68%% CI:    ${result.p68_low:.2f} – ${result.p68_high:.2f}",
            f"  95%% CI:    ${result.p95_low:.2f} – ${result.p95_high:.2f}",
            f"  Daily vol:  {result.daily_vol_pct:.2f}%%  "
            f"(ann {result.annualized_vol_pct:.1f}%%)",
            f"  ⚠️  Statistical baseline only — not a price prediction.",
        ]
        return "\n".join(lines)

    def _tradequality_cmd(args: list[str]) -> str:
        """Trade quality scoring from trade journal.

        Usage: /tradequality [N]
        Scores recent closed trades on outcome, hold time, risk-reward.
        Total 0-80 pts.  Grades: A≥70, B≥55, C≥40, D≥25, F<25
        """
        limit = 50
        if args and args[0].isdigit():
            limit = max(1, min(int(args[0]), 200))

        from amms.analysis.trade_quality import compute_quality

        try:
            report = compute_quality(conn, limit=limit)
        except Exception as e:
            return f"trade quality error: {e!r}"

        if report is None:
            return "No completed trades in journal to score."

        dist = report.grade_distribution
        dist_str = "  ".join(
            f"{g}:{dist.get(g, 0)}" for g in ("A", "B", "C", "D", "F")
        )

        lines = [
            f"── Trade Quality Report ({report.n_trades} trades) ──",
            f"  Avg score:  {report.avg_score:.1f}/80",
            f"  Grades:     {dist_str}",
        ]

        if report.best_trade:
            b = report.best_trade
            lines.append(
                f"  Best:       {b.symbol} [{b.grade}] "
                f"{b.total_score:.0f}pts  PnL ${b.pnl:+.2f}"
            )
        if report.worst_quality_trade:
            w = report.worst_quality_trade
            lines.append(
                f"  Worst:      {w.symbol} [{w.grade}] "
                f"{w.total_score:.0f}pts  PnL ${w.pnl:+.2f}"
            )

        lines.append("")
        lines.append("Recent trades:")
        for s in report.scores[:10]:
            hold_str = f"{s.hold_days:.1f}d" if s.hold_days is not None else "?d"
            lines.append(
                f"  [{s.grade}] {s.symbol:<6} {s.total_score:>4.0f}pt  "
                f"PnL {s.pnl_pct:+.1f}%%  {hold_str}"
            )

        return "\n".join(lines)

    def _drawdown_cmd(args: list[str]) -> str:
        """Portfolio equity drawdown from peak equity snapshot.

        Usage: /drawdown
        Shows peak equity, current drawdown, worst drawdown, and alert if severe.
        Reads from equity_snapshots table.
        """
        if conn is None:
            return "DB not wired."

        try:
            rows = conn.execute(
                "SELECT ts, equity FROM equity_snapshots ORDER BY ts ASC"
            ).fetchall()
        except Exception as e:
            return f"DB error: {e!r}"

        if not rows:
            return "No equity snapshots in DB."

        equities = [float(r[1]) for r in rows]
        peak = max(equities)
        current = equities[-1]
        current_dd = (current - peak) / peak * 100

        # Compute max drawdown (rolling peak)
        running_peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > running_peak:
                running_peak = eq
            dd = (eq - running_peak) / running_peak * 100
            if dd < max_dd:
                max_dd = dd

        lines = [
            "── Portfolio Equity Drawdown ──",
            f"  Peak equity:  ${peak:,.0f}",
            f"  Current:      ${current:,.0f}",
            f"  Current DD:   {current_dd:+.2f}%%",
            f"  worst (max):  {max_dd:+.2f}%%",
            f"  Snapshots:    {len(equities)}",
        ]
        if current_dd < -10.0:
            lines.append("  ⚠️  Drawdown exceeds 10%% — review risk exposure.")
        elif current_dd < -5.0:
            lines.append("  ⚠️  Drawdown exceeds 5%%.")

        return "\n".join(lines)

    def _posdd_cmd(args: list[str]) -> str:
        """Per-position price drawdown from recent peak.

        Usage: /posdd [LOOKBACK]
        Shows how far each position's price is from its recent peak.
        Default lookback: 60 bars.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions."

        lookback = 60
        if args and args[0].isdigit():
            lookback = max(5, min(int(args[0]), 120))

        bars_map: dict[str, list] = {}
        for pos in positions:
            try:
                bars = data.get_bars(pos.symbol, limit=lookback + 5)
                if bars:
                    bars_map[pos.symbol] = bars
            except Exception:
                pass

        from amms.analysis.drawdown_heatmap import analyze as dd_analyze

        heatmap = dd_analyze(bars_map, lookback=lookback)
        if heatmap is None:
            return "Could not compute drawdowns (insufficient data)."

        STATUS_ICON = {
            "new_high": "▲",
            "recovering": "↑",
            "stalling": "→",
            "deepening": "↓",
        }

        lines = [
            f"── Position Drawdown ({lookback}d lookback) ──",
            f"  Avg drawdown:  {heatmap.avg_drawdown_pct:+.1f}%%",
            f"  At new highs:  {heatmap.n_at_new_high}  "
            f"Deepening:  {heatmap.n_deepening}",
            "",
        ]
        for row in heatmap.rows:
            icon = STATUS_ICON.get(row.status, "?")
            lines.append(
                f"  {icon} {row.symbol:<6}  DD {row.drawdown_pct:+.1f}%%"
                f"  (max {row.max_drawdown_pct:+.1f}%%)"
                f"  rec {row.recovery_pct:.0f}%%"
                f"  {row.bars_since_peak}d ago"
            )
        return "\n".join(lines)

    def _volpct_cmd(args: list[str]) -> str:
        """Volatility percentile rank: where is current vol vs history?

        Usage: /volpct [SYM] [VOL_WINDOW] [HIST_WINDOW]
        Default: 20-day HV vs 252-day history.
        Regimes: low (≤25th), normal (≤60th), elevated (≤85th), extreme (>85th).
        """
        if data is None:
            return "Data client not wired."

        sym = None
        vol_window = 20
        hist_window = 252

        for arg in args:
            if arg.isdigit():
                v = int(arg)
                if v <= 60:
                    vol_window = v
                else:
                    hist_window = v
            else:
                sym = arg.upper()

        if sym is None:
            try:
                positions = broker.get_positions()
                if positions:
                    sym = positions[0].symbol
            except Exception:
                pass

        if sym is None:
            return "usage: /volpct [SYM] [VOL_WIN] [HIST_WIN]  (e.g. /volpct AAPL 20 252)"

        try:
            bars = data.get_bars(sym, limit=hist_window + vol_window + 10)
        except Exception as e:
            return f"data error for {sym}: {e!r}"

        from amms.analysis.vol_percentile import compute as vp_compute

        result = vp_compute(bars, vol_window=vol_window, history_window=hist_window)
        if result is None:
            return (
                f"Insufficient data for {sym} "
                f"(need {hist_window + vol_window}+ bars)."
            )

        REGIME_LABEL = {
            "low": "Low (compression)",
            "normal": "Normal",
            "elevated": "Elevated",
            "extreme": "Extreme (spike)",
        }

        lines = [
            f"── Vol Percentile: {sym} ──",
            f"  Current HV ({vol_window}d):  {result.realized_vol:.1f}%%",
            f"  Percentile:        {result.percentile:.1f}th  "
            f"/ {hist_window}d history",
            f"  Regime:            {REGIME_LABEL.get(result.regime, result.regime)}",
            f"  Mean HV:           {result.mean_vol:.1f}%%",
            f"  Median HV:         {result.median_vol:.1f}%%",
            f"  Vol-of-vol:        {result.vol_of_vol:.1f}%%",
        ]
        return "\n".join(lines)

    def _momentum_cmd(args: list[str]) -> str:
        """Momentum composite score aggregating RSI, ROC, MACD, Williams %R.

        Usage: /momentum [SYM]
        Single -100 (bearish) to +100 (bullish) score with per-indicator breakdown.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /momentum AAPL)"

        from amms.analysis.momentum_composite import compute as mc_compute

        SIGNAL_ICON = {
            "strong_bull": "▲▲",
            "bull": "▲",
            "neutral": "→",
            "bear": "▼",
            "strong_bear": "▼▼",
        }

        lines = ["── Momentum Composite ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = mc_compute(bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data (need 35+ bars)")
                continue

            icon = SIGNAL_ICON.get(result.signal, "?")
            lines.append(
                f"  {icon} {sym:<6}  score {result.score:+.0f}  [{result.signal}]"
            )

            def _fmt_c(v: float | None) -> str:
                return f"{v:+.1f}" if v is not None else "n/a"

            lines.append(
                f"     RSI {_fmt_c(result.rsi_component)}"
                f"  ROC {_fmt_c(result.roc_component)}"
                f"  MACD {_fmt_c(result.macd_component)}"
                f"  WR {_fmt_c(result.wr_component)}"
            )

        return "\n".join(lines)

    def _srlevels_cmd(args: list[str]) -> str:
        """Support and resistance levels from clustered pivot points.

        Usage: /srlevels [SYM]
        Identifies horizontal S/R zones with touch count and strength score.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        sym = args[0].upper() if args else (positions[0].symbol if positions else None)
        if sym is None:
            return "no open positions (pass a ticker: /srlevels AAPL)"

        try:
            bars = data.get_bars(sym, limit=70)
        except Exception as e:
            return f"data error for {sym}: {e!r}"

        from amms.analysis.support_resistance import detect as sr_detect

        result = sr_detect(bars)
        if result is None:
            return f"Insufficient data for {sym}."
        if not result.levels:
            return f"{sym}: no clear S/R levels detected."

        KIND_ICON = {"support": "S", "resistance": "R", "at_price": "≈"}

        lines = [
            f"── S/R Levels: {sym}  (current ${result.current_price:.2f}) ──",
        ]

        if result.nearest_resistance:
            r = result.nearest_resistance
            lines.append(
                f"  Nearest resistance: ${r.price:.2f}  "
                f"({r.distance_pct:+.1f}%%)  touches {r.touches}"
            )
        if result.nearest_support:
            s = result.nearest_support
            lines.append(
                f"  Nearest support:    ${s.price:.2f}  "
                f"({s.distance_pct:+.1f}%%)  touches {s.touches}"
            )

        lines.append("")
        lines.append("  All levels:")
        for lvl in reversed(result.levels):  # highest price first
            icon = KIND_ICON.get(lvl.kind, "?")
            lines.append(
                f"  [{icon}] ${lvl.price:.2f}  str {lvl.strength:.0f}  "
                f"({lvl.distance_pct:+.1f}%%)  {lvl.touches}x"
            )

        return "\n".join(lines)

    def _liquidity_cmd(args: list[str]) -> str:
        """Liquidity score: volume, spread, consistency, trend.

        Usage: /liquidity [SYM]
        Scores 0-100 (A-F) based on observable bar data.
        Low-liquidity symbols may have poor execution quality.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /liquidity AAPL)"

        from amms.analysis.liquidity_score import score as liq_score

        lines = ["── Liquidity Scores ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=35)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = liq_score(bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            WARN = "  ⚠️" if result.grade in ("D", "F") else ""
            lines.append(
                f"  [{result.grade}] {sym:<6}  {result.total_score:.0f}/100"
                f"  vol {result.avg_volume/1e3:.0f}k  "
                f"sprd {result.avg_spread_pct:.1f}%%  "
                f"{result.volume_trend}{WARN}"
            )

        return "\n".join(lines)

    def _regime_cmd(args: list[str]) -> str:
        """Market regime classification: trending/ranging, vol level.

        Usage: /regime [SYM]
        Classifies into: trending_up, trending_down, ranging_low_vol,
        ranging_high_vol.  Includes strategy hint for each regime.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /regime AAPL)"

        from amms.analysis.regime_classifier import classify as rc_classify

        REGIME_ICON = {
            "trending_up": "▲",
            "trending_down": "▼",
            "ranging_low_vol": "→",
            "ranging_high_vol": "↕",
        }
        REGIME_BIAS = {
            "trending_up": "BULL",
            "trending_down": "BEAR",
            "ranging_low_vol": "NEUTRAL",
            "ranging_high_vol": "NEUTRAL",
        }
        SIZE_MULTIPLIER = {
            "trending_up": 1.0,
            "trending_down": 0.5,
            "ranging_low_vol": 0.75,
            "ranging_high_vol": 0.5,
        }

        lines = ["── Market Regime ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=30)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = rc_classify(bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            icon = REGIME_ICON.get(result.regime, "?")
            bias = REGIME_BIAS.get(result.regime, "NEUTRAL")
            multiplier = SIZE_MULTIPLIER.get(result.regime, 1.0)
            lines.append(
                f"  {icon} {sym:<6}  {bias}  [{result.regime}]  "
                f"conf {result.confidence:.0f}%%"
            )
            lines.append(
                f"     trend {result.trend_strength:.0f}  "
                f"vol {result.vol_regime}  "
                f"mom {result.momentum_pct:+.1f}%%"
            )
            lines.append(
                f"     Size multiplier: ×{multiplier:.2f}"
            )
            lines.append(f"     → {result.strategy_hint}")

        return "\n".join(lines)

    def _heat_cmd(args: list[str]) -> str:
        """Position heat scores: composite rating for each open position.

        Usage: /heat
        Scores 0-100 on P&L, momentum, drawdown health, and liquidity.
        hot≥80, warm≥60, neutral≥40, cool≥20, cold<20.
        """
        if data is None:
            return "Data client not wired."

        from amms.analysis.position_heat import analyze as ph_analyze

        report = ph_analyze(broker, data)
        if report is None:
            return "No open positions."

        STATUS_ICON = {
            "hot": "🔥",
            "warm": "▲",
            "neutral": "→",
            "cool": "▼",
            "cold": "❄",
        }

        lines = [
            f"── Position Heat Report ──",
            f"  Avg score:  {report.avg_score:.0f}/100"
            f"  |  hot: {report.n_hot}  cold: {report.n_cold}",
            "",
        ]
        for ph in report.positions:
            icon = STATUS_ICON.get(ph.status, "?")
            lines.append(
                f"  {icon} {ph.symbol:<6}  {ph.score:.0f}pt [{ph.status}]"
                f"  PnL {ph.pnl_pct:+.1f}%%"
            )
            lines.append(
                f"     pnl {ph.pnl_score:.0f}"
                f"  mom {ph.momentum_score:.0f}"
                f"  DD {ph.drawdown_score:.0f}"
                f"  liq {ph.liquidity_score:.0f}"
            )

        return "\n".join(lines)

    def _concentration_cmd(args: list[str]) -> str:
        """Portfolio concentration risk via Herfindahl-Hirschman Index.

        Usage: /concentration
        Shows HHI, effective N, top-position weights, grade, and risk flags.
        """
        from amms.analysis.concentration_risk import analyze as cr_analyze

        report = cr_analyze(broker)
        if report is None:
            return "No open positions."

        lines = [
            f"── Concentration Risk ──",
            f"  Grade:       {report.grade}  — {report.verdict}",
            f"  HHI:         {report.hhi:.4f}  "
            f"(effective N: {report.effective_n:.1f})",
            f"  Positions:   {report.n_positions}",
            f"  Top 1:       {report.top1_pct:.1f}%%  "
            f"Top 3: {report.top3_pct:.1f}%%  "
            f"Top 5: {report.top5_pct:.1f}%%",
            "",
        ]

        lines.append("  Holdings by weight:")
        for pw in report.positions:
            bar = "█" * int(pw.weight_pct / 5)
            lines.append(f"  {pw.symbol:<6}  {pw.weight_pct:>5.1f}%%  {bar}")

        if report.risk_flags:
            lines.append("")
            lines.append("  ⚠️  Risk flags:")
            for flag in report.risk_flags:
                lines.append(f"    • {flag}")

        return "\n".join(lines)

    def _breakout_cmd(args: list[str]) -> str:
        """Breakout and squeeze detection for open positions.

        Usage: /breakout [SYM]
        Detects range compression (squeeze) and volume-confirmed breakouts.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /breakout AAPL)"

        from amms.analysis.breakout_detector import detect as bo_detect

        SIGNAL_ICON = {
            "breakout_up": "▲▲",
            "breakout_down": "▼▼",
            "squeeze": "⚡",
            "none": "—",
        }

        lines = ["── Breakout / Squeeze Detection ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=30)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = bo_detect(bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            icon = SIGNAL_ICON.get(result.signal, "?")
            conf_str = f"{result.confidence:.0f}%%" if result.confidence > 0 else ""
            lvl_str = (f"  level ${result.breakout_level:.2f}"
                       if result.breakout_level else "")
            lines.append(
                f"  {icon} {sym:<6}  [{result.signal}]  {conf_str}"
                f"  vol×{result.volume_ratio:.1f}"
                f"  compress {result.range_compression:.2f}"
                f"{lvl_str}"
            )

        return "\n".join(lines)

    def _dashboard_cmd(args: list[str]) -> str:
        """Portfolio dashboard: compact summary of key metrics.

        Usage: /dashboard
        Combines account equity, position heat, concentration risk,
        regime, and breakout signals into one overview.
        """
        lines = ["══ Portfolio Dashboard ══"]

        # 1. Account summary
        try:
            account = broker.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            lines.append(f"  Equity:       ${equity:,.0f}")
            lines.append(f"  Buying power: ${buying_power:,.0f}")
        except Exception:
            lines.append("  Account: (unavailable)")

        # 2. Positions summary
        try:
            positions = broker.get_positions()
            n_pos = len(positions)
            total_mv = sum(float(p.market_value) for p in positions)
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            pl_sign = "+" if total_pl >= 0 else ""
            lines.append(f"  Positions:    {n_pos}  MV ${total_mv:,.0f}  "
                         f"P&L {pl_sign}${total_pl:,.0f}")
        except Exception:
            positions = []
            lines.append("  Positions: (unavailable)")

        lines.append("")

        # 3. Concentration risk
        from amms.analysis.concentration_risk import analyze as cr_analyze
        cr = cr_analyze(broker)
        if cr:
            lines.append(f"  Concentration [{cr.grade}]:  HHI {cr.hhi:.3f}"
                         f"  eff-N {cr.effective_n:.1f}"
                         f"  top1 {cr.top1_pct:.0f}%%")
        else:
            lines.append("  Concentration: n/a")

        # 4. Position heat (avg)
        if data and positions:
            from amms.analysis.position_heat import analyze as ph_analyze
            heat_report = ph_analyze(broker, data)
            if heat_report:
                lines.append(
                    f"  Heat avg:     {heat_report.avg_score:.0f}/100"
                    f"  hot {heat_report.n_hot}  cold {heat_report.n_cold}"
                )
            else:
                lines.append("  Heat: n/a")

        # 5. Regime for top position
        if data and positions:
            from amms.analysis.regime_classifier import classify as rc_classify
            top_sym = positions[0].symbol
            try:
                rbars = data.get_bars(top_sym, limit=30)
                rr = rc_classify(rbars)
                if rr:
                    lines.append(
                        f"  Regime ({top_sym}): [{rr.regime}]  "
                        f"conf {rr.confidence:.0f}%%"
                    )
            except Exception:
                pass

        lines.append("")
        lines.append("  /heat  /concentration  /rsrank  /regime  /breakout")
        return "\n".join(lines)

    def _targets_cmd(args: list[str]) -> str:
        """Profit target tracker: 1R/2R/3R targets for open positions.

        Usage: /targets [SYM]
        Shows ATR-based profit targets and current progress (R-multiple).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions."

        if args:
            sym_filter = args[0].upper()
            positions = [p for p in positions if p.symbol == sym_filter]
            if not positions:
                return f"{sym_filter} not in open positions."

        from amms.analysis.profit_target import compute as pt_compute

        lines = ["── Profit Targets (ATR-based) ──"]
        for pos in positions[:8]:
            sym = pos.symbol
            try:
                entry = float(pos.avg_entry_price)
            except Exception:
                entry = 0.0
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = pt_compute(sym, entry, bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            exceeded = " ✓" if result.exceeded_2r else ""
            bar_w = max(0, min(20, int(result.pct_to_2r / 5)))
            progress_bar = "█" * bar_w + "░" * (20 - bar_w)
            lines.append(
                f"  {sym:<6}  {result.r_multiple:+.2f}R  "
                f"PnL {result.pnl_pct:+.1f}%%{exceeded}"
            )
            lines.append(
                f"    [{progress_bar}] {result.pct_to_2r:.0f}%% to 2R"
            )
            lines.append(
                f"    stop ${result.stop_1atr:.2f}"
                f"  1R ${result.target_1r:.2f}"
                f"  2R ${result.target_2r:.2f}"
                f"  3R ${result.target_3r:.2f}"
            )

        return "\n".join(lines)

    def _kelly_cmd(args: list[str]) -> str:
        """Kelly Criterion position sizer.

        Usage: /kelly CAPITAL RISK_PCT WIN_RATE AVG_WIN AVG_LOSS [PRICE]
          CAPITAL   — total portfolio value ($)
          RISK_PCT  — % of capital you risk per trade (e.g. 2.0)
          WIN_RATE  — historical win rate as fraction (e.g. 0.55)
          AVG_WIN   — average winning trade PnL ($)
          AVG_LOSS  — average losing trade PnL (positive, $)
          PRICE     — current share price (optional, for share count)

        Example: /kelly 10000 2.0 0.55 300 150 50.00
        """
        USAGE = (
            "Usage: /kelly CAPITAL RISK_PCT WIN_RATE AVG_WIN AVG_LOSS [PRICE]\n"
            "  Example: /kelly 10000 2.0 0.55 300 150 50.00"
        )

        if not args or len(args) < 5:
            # Fallback to DB-based if conn is available and no args given
            if not args and conn is not None:
                from amms.analysis.kelly_sizer import compute as kelly_compute
                try:
                    account = broker.get_account()
                    portfolio_value = float(account.portfolio_value)
                except Exception:
                    portfolio_value = None
                result = kelly_compute(conn, portfolio_value=portfolio_value)
                if result is not None:
                    grade_icon = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}.get(result.grade, "")
                    lines = [
                        f"── Kelly Criterion Sizer ({result.n_trades} trades) ──",
                        f"  Win rate:      {result.win_rate:.1f}%%  ({result.n_wins}W / {result.n_losses}L)",
                        f"  Avg win:       ${result.avg_win:.2f}",
                        f"  Avg loss:      ${result.avg_loss:.2f}",
                        f"  Payoff ratio:  {result.payoff_ratio:.2f}×",
                        f"  Edge grade:    {grade_icon} {result.grade}",
                        "",
                        f"  Full Kelly:    {result.kelly_pct:.1f}%%",
                        f"  Half Kelly:    {result.half_kelly_pct:.1f}%%  ← recommended",
                        f"  Qtr Kelly:     {result.quarter_kelly_pct:.1f}%%",
                    ]
                    if result.suggested_value is not None:
                        lines.append(f"  Suggested:     ${result.suggested_value:,.0f}")
                    lines.append(f"  {result.note}")
                    return "\n".join(lines)
            return USAGE

        try:
            capital = float(args[0])
            risk_pct = float(args[1])
            win_rate = float(args[2])
            avg_win = float(args[3])
            avg_loss = float(args[4])
        except (ValueError, IndexError):
            return "Invalid input — " + USAGE

        price = None
        if len(args) >= 6:
            try:
                price = float(args[5])
            except ValueError:
                pass

        if not (0 < win_rate < 1):
            return "WIN_RATE must be between 0 and 1 (e.g. 0.55 for 55%)."
        if avg_loss <= 0 or avg_win <= 0:
            return "AVG_WIN and AVG_LOSS must be positive values."

        payoff_ratio = avg_win / avg_loss
        kelly_raw = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        kelly_raw = max(0.0, kelly_raw)
        kelly_pct = min(kelly_raw * 100, 25.0)
        half_kelly_pct = kelly_pct * 0.5
        quarter_kelly_pct = kelly_pct * 0.25

        risk_amount = capital * risk_pct / 100

        half_kelly_alloc = capital * half_kelly_pct / 100
        shares = int(half_kelly_alloc / price) if price and price > 0 else None

        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        lines = [
            f"── Kelly Criterion ──",
            f"  Capital:       ${capital:,.2f}",
            f"  Win rate:      {win_rate*100:.1f}%%",
            f"  Avg win/loss:  ${avg_win:.2f} / ${avg_loss:.2f}",
            f"  Payoff ratio:  {payoff_ratio:.2f}×",
            f"  Edge:          ${edge:+.2f} per trade",
            "",
            f"  Full Kelly:    {kelly_pct:.1f}%%  = ${capital*kelly_pct/100:,.2f}",
            f"  Half Kelly:    {half_kelly_pct:.1f}%%  = ${half_kelly_alloc:,.2f}  ← recommended",
            f"  Qtr Kelly:     {quarter_kelly_pct:.1f}%%  = ${capital*quarter_kelly_pct/100:,.2f}",
        ]
        if shares is not None:
            lines.append(f"  Shares (half-K, @${price:.2f}): {shares}")
        if kelly_pct == 0:
            lines.append("")
            lines.append("  No edge — skip this trade setup.")
        return "\n".join(lines)

    def _beta_cmd(args: list[str]) -> str:
        """Portfolio beta vs SPY (market sensitivity).

        Usage: /beta [BENCHMARK]  (default SPY)
        Shows each position's beta, alpha, R², and weighted portfolio beta.
        beta>1 = amplifies market moves, beta<1 = defensive.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions."

        benchmark = args[0].upper() if args else "SPY"

        from amms.analysis.beta_calculator import compute as beta_compute

        # Fetch benchmark bars
        try:
            bench_bars = data.get_bars(benchmark, limit=70)
        except Exception as e:
            return f"Could not fetch benchmark {benchmark}: {e!r}"

        # Fetch position bars
        positions_bars: dict[str, list] = {}
        for pos in positions[:10]:
            sym = pos.symbol
            try:
                positions_bars[sym] = data.get_bars(sym, limit=70)
            except Exception:
                pass

        result = beta_compute(positions_bars, bench_bars, benchmark=benchmark)
        if result is None:
            return f"Not enough data to compute beta vs {benchmark}."

        beta_icon = lambda b: "🔴" if b > 1.5 else ("🟡" if b > 1.1 else ("🟢" if b < 0.7 else "↔️ "))

        lines = [
            f"── Portfolio Beta vs {result.benchmark} ({result.n_positions} positions) ──",
            f"  Portfolio β:  {result.portfolio_beta:.3f}",
            f"  Avg R²:       {result.avg_r_squared:.3f}",
            f"  High-β (>1.5): {result.high_beta_count}  |  Defensive (<0.7): {result.defensive_count}",
            f"  {result.verdict}",
            "",
            f"  {'Symbol':<7}  {'β':>6}  {'α ann%%':>8}  {'R²':>5}  {'corr':>6}  {'wt%%':>5}",
        ]
        for r in result.positions:
            icon = beta_icon(r.beta)
            lines.append(
                f"  {r.symbol:<7}  {icon} {r.beta:>5.3f}  {r.alpha_annualized:>+7.1f}%%"
                f"  {r.r_squared:>5.3f}  {r.correlation:>+5.3f}  {r.weight_pct:>4.1f}%%"
            )

        return "\n".join(lines)

    def _sigagg_cmd(args: list[str]) -> str:
        """Multi-indicator signal aggregator: RSI, MACD, ROC, WR, SMA, Stoch, OBV.

        Usage: /sigagg [SYM]
        Votes from 8 indicators aggregated into -100 to +100 consensus score.
        Positive = bullish, negative = bearish. Shows per-indicator votes.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /sigagg AAPL)"

        from amms.analysis.signal_aggregator import compute as sa_compute

        signal_icon = {
            "strong_bull": "🚀", "bull": "🟢",
            "neutral": "↔️ ", "bear": "🔴", "strong_bear": "⛔",
        }
        vote_icon = {1: "🟢", 0: "↔️ ", -1: "🔴"}

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = sa_compute(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 35+ bars)")
                continue

            icon = signal_icon.get(result.signal, "")
            lines += [
                f"── {sym} Signal Consensus ──",
                f"  Score:  {result.score:+.0f}/100  {icon} {result.signal}",
                f"  Votes:  {result.bull_votes}🟢  {result.bear_votes}🔴  {result.neutral_votes}↔️",
                "",
            ]
            for v in result.votes:
                vi = vote_icon.get(v.vote, "")
                val_str = f"{v.value:.2f}" if v.value is not None else "n/a"
                lines.append(f"  {vi} {v.name:<14}  {val_str:>8}  [{v.signal}]")
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _trade_timing_cmd(args: list[str]) -> str:
        """Trade timing analysis: best day-of-week and hour-of-day to enter.

        Usage: /timing [LIMIT]  (default 500 trades)
        Shows win rate and avg PnL% per weekday and per hour of entry.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(20, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.trade_timing import compute as tt_compute

        result = tt_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ trades with buy timestamps)."

        lines = [
            f"── Trade Timing Analysis ({result.n_trades} trades) ──",
            "",
            "By Weekday:",
        ]
        for b in result.by_weekday:
            bar = "█" * int(max(0, b.avg_pnl_pct) * 5) if b.avg_pnl_pct > 0 else "░" * int(abs(min(0, b.avg_pnl_pct)) * 5)
            mark = " ◀ best" if b.label == result.best_weekday else (" ◀ worst" if b.label == result.worst_weekday else "")
            lines.append(
                f"  {b.label:<10}  {b.n_trades:>3}tr  "
                f"WR:{b.win_rate:.0f}%  avg:{b.avg_pnl_pct:+.2f}%{mark}"
            )

        if result.by_hour:
            lines += ["", "By Hour (top 5):"]
            sorted_hours = sorted(result.by_hour, key=lambda b: b.avg_pnl_pct, reverse=True)
            for b in sorted_hours[:5]:
                mark = " ◀ best" if b.label == result.best_hour else ""
                lines.append(
                    f"  {b.label}  {b.n_trades:>3}tr  "
                    f"WR:{b.win_rate:.0f}%  avg:{b.avg_pnl_pct:+.2f}%{mark}"
                )

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _ribbon_cmd(args: list[str]) -> str:
        """Moving Average Ribbon: 6 EMAs ordered = strong trend, tangled = ranging.

        Usage: /ribbon [SYM]  (default: all open positions)
        Shows EMA-5/8/13/21/34/55 alignment and ribbon expansion/contraction.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /ribbon AAPL)"

        from amms.analysis.ma_ribbon import analyze as ribbon_analyze

        dir_icon = {"up": "↑ UPTREND", "down": "↓ DOWNTREND", "tangled": "↔ RANGING"}
        expand_icon = {True: "📈 expanding", False: "📉 contracting"}

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = ribbon_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 60+ bars)")
                continue

            di = dir_icon.get(result.direction, result.direction)
            lines += [
                f"── {sym} MA Ribbon ──",
                f"  Direction:  {di}  ({'ordered' if result.ordered else 'tangled'})",
                f"  Price:      {result.current_price:.2f}  [{result.price_position.replace('_', ' ')}]",
                f"  Spread:     {result.ribbon_spread_pct:.2f}%  {expand_icon.get(result.is_expanding, '')}",
                "",
            ]
            for p in result.periods:
                v = result.emas[p]
                diff = (result.current_price - v) / v * 100
                lines.append(f"  EMA-{p:<3}  {v:>9.2f}  ({diff:+.2f}% from price)")
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _fvg_cmd(args: list[str]) -> str:
        """Fair Value Gap detector: find price imbalance zones.

        Usage: /fvg [SYM]  (default: all open positions)
        Identifies three-candle patterns where price left a visible gap
        (institutional order imbalance zone) that may act as future magnet.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /fvg AAPL)"

        from amms.analysis.fair_value_gap import detect as fvg_detect

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = fvg_detect(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars)")
                continue

            lines += [
                f"── {sym} Fair Value Gaps ──",
                f"  Price: {result.current_price:.2f}  |  {result.bars_scanned} bars scanned",
                f"  Found: {len(result.fvgs)} gaps  ({len(result.active_fvgs)} active, "
                f"{result.bullish_count} bull, {result.bearish_count} bear)",
                "",
            ]

            if result.nearest_bearish_gap:
                g = result.nearest_bearish_gap
                lines.append(f"  ↓ Nearest bear FVG (resistance): {g.lower:.2f}–{g.upper:.2f}  ({g.size_pct:.2f}%)")
            lines.append(f"  — Price: {result.current_price:.2f} —")
            if result.nearest_bullish_gap:
                g = result.nearest_bullish_gap
                lines.append(f"  ↑ Nearest bull FVG (support): {g.lower:.2f}–{g.upper:.2f}  ({g.size_pct:.2f}%)")

            # Show last 5 active FVGs
            active_recent = result.active_fvgs[:5]
            if active_recent:
                lines += ["", "  Active gaps (newest first):"]
                for g in active_recent:
                    icon = "🟢" if g.kind == "bullish" else "🔴"
                    pfx = "▲" if g.kind == "bullish" else "▼"
                    lines.append(f"    {icon} {pfx} {g.lower:.2f}–{g.upper:.2f}  ({g.size_pct:.2f}%)")
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _pivots_cmd(args: list[str]) -> str:
        """Pivot points: classic, Fibonacci, or Camarilla support/resistance levels.

        Usage: /pivots [SYM] [method]  (method: classic/fib/cam, default classic)
        Computes pivot levels from yesterday's H/L/C for all open positions.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        # Parse args: /pivots [SYM] [method]
        symbols = []
        method = "classic"
        for a in args:
            if a.lower() in ("classic", "fibonacci", "fib", "camarilla", "cam"):
                method = {"fib": "fibonacci", "cam": "camarilla"}.get(a.lower(), a.lower())
            else:
                symbols.append(a.upper())
        if not symbols:
            symbols = [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /pivots AAPL)"

        from amms.analysis.pivot_points import compute as pp_compute

        kind_label = {"R": "R", "S": "S", "PP": "◆"}
        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=5)
            except Exception:
                bars = []
            if not bars or len(bars) < 2:
                lines.append(f"  {sym}  n/a (need bar data)")
                continue
            # Use second-to-last bar as "previous period"
            prev = bars[-2]
            curr_price = float(bars[-1].close)
            result = pp_compute(
                float(prev.high), float(prev.low), float(prev.close),
                curr_price, symbol=sym, method=method,
            )
            if result is None:
                lines.append(f"  {sym}  error computing pivots")
                continue

            lines += [f"── {sym} Pivots ({method}) ──"]
            for lv in reversed(result.levels):
                arrow = " ← price" if abs(lv.price - curr_price) / curr_price < 0.005 else ""
                label = kind_label.get(lv.kind, lv.kind)
                lines.append(f"  {label}{lv.name:<3}  {lv.price:>9.2f}{arrow}")
            lines += [f"  Current: {curr_price:.2f}  [{result.current_zone}]", ""]

        return "\n".join(lines).rstrip()

    def _squeeze_cmd(args: list[str]) -> str:
        """Volatility squeeze / price compression detector.

        Usage: /squeeze [SYM]  (default: all open positions)
        Detects when ATR and Bollinger Band width are contracting below their
        historical average — a classic pre-breakout setup.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /squeeze AAPL)"

        from amms.analysis.price_compression import analyze as pc_analyze

        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = pc_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 35+ bars)")
                continue

            squeeze_icon = "⚡" if result.compressed else ("⚠️" if result.atr_ratio < 0.75 else "✅")
            lines += [
                f"── {sym} Compression ──",
                f"  Status:    {squeeze_icon} {'SQUEEZE' if result.compressed else 'normal'}  "
                f"(strength {result.compression_strength:.0f}/100)",
                f"  ATR ratio:  {result.atr_ratio:.2f}  ({result.atr_current:.3f} vs avg {result.atr_average:.3f})",
                f"  BB width:  {result.bb_width_ratio:.2f}  ({result.bb_width_current:.2f}% vs avg {result.bb_width_average:.2f}%)",
                f"  Range:     {result.range_low:.2f} – {result.range_high:.2f}",
                f"  Bias:      {result.bias}",
                f"  Squeeze:   {result.bars_in_squeeze} bars",
                f"  {result.verdict}",
                "",
            ]
        return "\n".join(lines).rstrip()

    def _mstruct_cmd(args: list[str]) -> str:
        """Market structure: HH/HL uptrend vs LH/LL downtrend, CHoCH detection.

        Usage: /mstruct [SYM]  (default: all open positions)
        Identifies swing highs/lows and classifies trend structure.
        Flags Change of Character (CHoCH) when structure breaks.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /mstruct AAPL)"

        from amms.analysis.market_structure import analyze as ms_analyze

        struct_icon = {
            "uptrend": "↑↑", "downtrend": "↓↓", "ranging": "↔↔", "unclear": "??"
        }
        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = ms_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need more bars)")
                continue

            si = struct_icon.get(result.structure, "")
            choch_note = f"  ⚠️ CHoCH ({result.choch_direction})!" if result.choch_detected else ""
            lines += [
                f"── {sym} Market Structure ──",
                f"  Structure: {si} {result.structure.upper()}{choch_note}",
                f"  Price: {result.current_price:.2f}",
            ]
            if result.last_swing_high:
                lines.append(f"  Last swing high: {result.last_swing_high:.2f}  ({result.pct_from_last_high:+.1f}% from price)")
            if result.last_swing_low:
                lines.append(f"  Last swing low:  {result.last_swing_low:.2f}  ({result.pct_from_last_low:+.1f}% from price)")
            if result.swing_highs:
                highs_str = " → ".join(f"{h:.2f}" for h in result.swing_highs)
                lines.append(f"  Swing highs: {highs_str}")
            if result.swing_lows:
                lows_str = " → ".join(f"{l:.2f}" for l in result.swing_lows)
                lines.append(f"  Swing lows:  {lows_str}")
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _mtf_cmd(args: list[str]) -> str:
        """Multi-timeframe trend analysis: short/medium/long alignment.

        Usage: /mtf [SYM]  (default: all open positions)
        Derives short, medium, and long-term trends from the same bar series
        and shows whether all timeframes are aligned.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /mtf AAPL)"

        from amms.analysis.multi_timeframe import analyze as mtf_analyze

        dir_icon = {"up": "↑", "down": "↓", "flat": "↔"}
        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = mtf_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 20+ bars)")
                continue

            align_icon = "✅" if result.aligned else "⚠️"
            lines += [
                f"── {sym} Multi-Timeframe ──",
                f"  Price: {result.current_price:.2f}  |  {result.bars_available} bars",
                f"  Alignment: {align_icon} {result.alignment_score:.0f}%  [{result.dominant_direction.upper()}]",
                "",
            ]
            for t in result.tiers:
                di = dir_icon.get(t.direction, "")
                lines.append(
                    f"  {t.label:<8}  {di} {t.direction:<5}  "
                    f"EMA={t.ema:.2f}  price {t.price_vs_ema}  "
                    f"slope={t.slope_pct:+.3f}%/bar"
                )
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _heikin_ashi_cmd(args: list[str]) -> str:
        """Heikin-Ashi trend analysis: smoothed candles, uptrend/downtrend strength.

        Usage: /ha [SYM]  (default: all open positions)
        Converts standard candles to Heikin-Ashi and identifies trend direction,
        consecutive run length, and no-wick (high-conviction) candle count.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /ha AAPL)"

        from amms.analysis.heikin_ashi import analyze as ha_analyze

        trend_icon = {"up": "↑", "down": "↓", "reversal": "⚡", "consolidating": "↔"}
        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = ha_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars)")
                continue

            ti = trend_icon.get(result.trend, "")
            run_str = (
                f"{result.consecutive_bull} bull" if result.consecutive_bull
                else f"{result.consecutive_bear} bear"
            )
            strong_note = f"  ({result.strong_candles} no-wick)" if result.strong_candles else ""
            lines += [
                f"── {sym} Heikin-Ashi ({result.bars_used} bars) ──",
                f"  Trend:    {ti} {result.trend.upper()}  (strength {result.trend_strength:.0f}/100)",
                f"  Run:      {run_str} consecutive{strong_note}",
                f"  HA open:  {result.current_ha_open:.2f}  |  HA close: {result.current_ha_close:.2f}",
                f"  {result.verdict}",
                "",
            ]

        return "\n".join(lines).rstrip()

    def _avwap_cmd(args: list[str]) -> str:
        """Anchored VWAP: VWAP from a swing high/low or custom bar offset.

        Usage: /avwap [SYM] [anchor]
          anchor: "high" (swing high), "low" (swing low, default), or a number
                  of bars back (e.g. /avwap AAPL 20)
        Shows AVWAP, ±1σ/±2σ bands, and price position relative to AVWAP.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        # Parse args: /avwap [SYM] [anchor]
        sym_arg = None
        anchor_arg: str | int = "auto_low"
        for a in args:
            try:
                anchor_arg = int(a)
            except ValueError:
                if a.lower() in ("high", "auto_high"):
                    anchor_arg = "auto_high"
                elif a.lower() in ("low", "auto_low"):
                    anchor_arg = "auto_low"
                elif a.upper().isalpha():
                    sym_arg = a.upper()

        symbols = [sym_arg] if sym_arg else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /avwap AAPL)"

        from amms.analysis.anchored_vwap import analyze as avwap_analyze

        pos_icon = {"above": "↑", "below": "↓", "at": "↔"}
        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=100)
            except Exception:
                bars = []
            result = avwap_analyze(bars, symbol=sym, anchor=anchor_arg)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars with volume)")
                continue

            pi = pos_icon.get(result.price_position, "")
            lines += [
                f"── {sym} Anchored VWAP ({result.anchor_label}) ──",
                f"  AVWAP:    {result.avwap:.2f}  |  Price: {result.current_price:.2f}  "
                f"{pi} {result.pct_from_avwap:+.2f}%",
                f"  +2σ/+1σ:  {result.upper_2:.2f} / {result.upper_1:.2f}",
                f"  -1σ/-2σ:  {result.lower_1:.2f} / {result.lower_2:.2f}",
                f"  Window:   {result.bars_in_window} bars  (of {result.total_bars})",
                f"  {result.verdict}",
                "",
            ]

        return "\n".join(lines).rstrip()

    def _ofi_cmd(args: list[str]) -> str:
        """Order Flow Imbalance: estimates buy vs. sell pressure from OHLCV bars.

        Usage: /ofi [SYM]  (default: all open positions)
        Uses close-position within bar range as a buy/sell pressure proxy.
        Cumulative OFI trend + divergence from price = accumulation/distribution signal.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /ofi AAPL)"

        from amms.analysis.order_flow import analyze as ofi_analyze

        dir_icon = {"rising": "↑", "falling": "↓", "flat": "↔"}
        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = ofi_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 5+ bars)")
                continue

            di = dir_icon.get(result.cofi_direction, "")
            div_note = "  ⚠️ DIVERGENCE" if result.divergence else ""
            lines += [
                f"── {sym} Order Flow ({result.bars_used} bars) ──",
                f"  COFI:     {result.cofi:+.3f}  {di} {result.cofi_direction}{div_note}",
                f"  Avg OFI:  {result.avg_ofi:+.4f}/bar",
                f"  Buy pres: {result.buy_pressure_pct:.0f}% of bars",
                f"  Price:    {result.current_price:.2f}  ({result.price_trend})",
                f"  {result.verdict}",
                "",
            ]

        return "\n".join(lines).rstrip()

    def _volskew_cmd(args: list[str]) -> str:
        """Realized volatility skew: upside vs. downside vol, tail ratio, gain-to-pain.

        Usage: /volskew [SYM]  (default: all open positions)
        Negative skew (down vol > up vol) signals crash risk.
        Positive skew (up vol > down vol) signals breakout potential.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /volskew AAPL)"

        from amms.analysis.vol_skew import analyze as vs_analyze

        skew_icon = {"positive": "↑", "negative": "↓", "symmetric": "↔"}
        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=120)
            except Exception:
                bars = []
            result = vs_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 10+ bars with up+down returns)")
                continue

            si = skew_icon.get(result.skew_label, "")
            lines += [
                f"── {sym} Vol Skew ({result.bars_used} bars) ──",
                f"  Skew:     {si} {result.skew_label.upper()}  ({result.skew:+.3f})",
                f"  RV total: {result.rv_total:.1f}%  "
                f"(up {result.rv_up:.1f}% / down {result.rv_down:.1f}%)",
                f"  Tail ratio:   {result.tail_ratio:.2f}  {'(fat upside)' if result.tail_ratio > 1.2 else '(fat downside)' if result.tail_ratio < 0.8 else '(balanced)'}",
                f"  Gain/pain:    {result.gain_to_pain:.2f}",
                f"  Semi-dev:     {result.semi_deviation:.1f}%",
                f"  Up days:      {result.up_days_pct:.0f}%  "
                f"(avg +{result.avg_up_return:.2f}% / avg {result.avg_down_return:.2f}%)",
                f"  {result.verdict}",
                "",
            ]

        return "\n".join(lines).rstrip()

    def _ptconsensus_cmd(args: list[str]) -> str:
        """Price Target Consensus: clusters Fib/ATR/Pivot/Measured targets.

        Usage: /ptc [SYM]  (default: all open positions)
        Derives upside and downside targets from multiple technical methods,
        clusters nearby targets into consensus zones, and shows the strongest zone.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /ptc AAPL)"

        from amms.analysis.price_target_consensus import analyze as ptc_analyze

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = ptc_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 20+ bars)")
                continue

            lines += [f"── {sym} Price Targets ({result.bars_used}b) ──"]
            lines.append(f"  Current: {result.current_price:.2f}")

            if result.best_upside_zone:
                z = result.best_upside_zone
                sources_str = " / ".join(t.source for t in z.targets[:3])
                lines += [
                    f"  ↑ Upside zone:   {z.low:.2f} – {z.high:.2f}  "
                    f"(center {z.center:.2f}, +{z.pct_from_current:.1f}%)",
                    f"    Sources: {sources_str}  [{z.n_sources} methods, "
                    f"consensus {result.upside_consensus_pct:.0f}%]",
                ]
            else:
                lines.append("  ↑ No clear upside consensus")

            if result.best_downside_zone:
                z = result.best_downside_zone
                sources_str = " / ".join(t.source for t in z.targets[:3])
                lines += [
                    f"  ↓ Downside zone: {z.low:.2f} – {z.high:.2f}  "
                    f"(center {z.center:.2f}, {z.pct_from_current:.1f}%)",
                    f"    Sources: {sources_str}  [{z.n_sources} methods, "
                    f"consensus {result.downside_consensus_pct:.0f}%]",
                ]
            else:
                lines.append("  ↓ No clear downside consensus")

            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _regime_sizer_cmd(args: list[str]) -> str:
        """Regime-conditioned position sizer: Kelly adjusted for market environment.

        Usage: /regsize [SYM]  (default: first open position)
        Reads historical win rate from trade DB, detects current ATR/trend regime,
        and outputs Kelly fraction scaled by regime multiplier (0.10× – 1.00×).
        """
        if conn is None:
            return "DB not wired."
        if data is None:
            return "Data client not wired."

        # Pull historical stats from trade DB
        try:
            rows = conn.execute(
                "SELECT pnl_pct FROM trades WHERE status='closed' AND pnl_pct IS NOT NULL "
                "ORDER BY closed_at DESC LIMIT 300"
            ).fetchall()
        except Exception as e:
            return f"DB error: {e!r}"

        if len(rows) < 10:
            return "Need 10+ closed trades for regime sizing."

        pnl_pcts = [float(r[0]) for r in rows]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p < 0]
        if not wins or not losses:
            return "Need both wins and losses in trade history."

        win_rate = len(wins) / len(pnl_pcts) * 100.0
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        # Get bars for the requested symbol or first open position
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        sym = args[0].upper() if args else (positions[0].symbol if positions else None)
        if sym is None:
            return "no open positions (pass a ticker: /regsize AAPL)"

        try:
            bars = data.get_bars(sym, limit=60)
        except Exception:
            bars = []

        try:
            port_val = sum(float(p.market_value) for p in positions) if positions else None
            cur_price = float(bars[-1].close) if bars else None
        except Exception:
            port_val = None
            cur_price = None

        from amms.analysis.regime_sizer import analyze as rs_analyze

        result = rs_analyze(
            bars,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            portfolio_value=port_val,
            current_price=cur_price,
        )
        if result is None:
            return f"Could not compute regime size for {sym} (need 5+ bars)."

        regime_icon = {
            "calm_bull": "🟢", "calm_neutral": "🟡", "calm_bear": "🟠",
            "hot_bull": "🔶", "hot_neutral": "🔴", "hot_bear": "💀",
            "extreme_vol": "🚨",
        }.get(result.regime, "")

        lines = [
            f"── {sym} Regime Sizer ──",
            f"  Regime:     {regime_icon} {result.regime}  (×{result.regime_multiplier:.2f})",
            f"  ATR:        {result.atr_pct:.2f}%/bar  |  Trend: {result.trend_direction}",
            "",
            f"  Win rate:   {result.win_rate:.1f}%  |  Payoff: {result.payoff_ratio:.2f}×",
            f"  Full Kelly: {result.kelly_pct:.1f}%",
            f"  Half Kelly: {result.half_kelly_pct:.1f}%",
            f"  Adj. size:  {result.adjusted_pct:.2f}% of capital",
            f"  Max loss/trade: {result.max_loss_pct:.2f}%",
        ]
        if result.suggested_shares is not None:
            lines.append(f"  Suggested:  {result.suggested_shares} shares")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _tcluster_cmd(args: list[str]) -> str:
        """Trade clustering: time patterns, burst trading, symbol concentration.

        Usage: /tcluster [LIMIT]  (default 500 trades)
        Detects whether you trade in bursts, always at the same hour,
        over-concentrate in one stock, or anchor to round price levels.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.trade_clustering import compute as tc_compute

        result = tc_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ closed trades)."

        lines = [f"── Trade Clustering ({result.n_trades} trades) ──", ""]

        if result.top_hours:
            lines.append("  Time Distribution (top hours):")
            for h in result.top_hours:
                bar = "█" * min(20, int(h.n_trades / result.n_trades * 40))
                lines.append(f"    {h.hour:02d}:00  {bar} {h.n_trades}× | "
                             f"WR {h.win_rate:.0f}% | avg {h.avg_pnl_pct:+.2f}%")
            lines.append(f"  Hour concentration: {result.hour_concentration:.2f}  "
                         f"(>0.5 = clustered)")
            lines.append("")

        if result.burst_windows:
            lines.append(f"  Burst windows: {len(result.burst_windows)}  "
                         f"({result.n_burst_trades} trades, {result.burst_pct:.0f}% of total)")
            lines.append(f"  Burst avg PnL: {result.burst_avg_pnl:+.2f}%  vs  "
                         f"normal: {result.non_burst_avg_pnl:+.2f}%")
            lines.append("")

        if result.top_symbols:
            lines.append("  Symbol concentration (top 5):")
            for sym, cnt in result.top_symbols:
                pct = cnt / result.n_trades * 100
                lines.append(f"    {sym:<8}  {cnt}× ({pct:.0f}%)")
            lines.append(f"  Symbol HHI: {result.symbol_concentration:.2f}  (>0.25 = concentrated)")
            lines.append("")

        lines.append(f"  Round-number entries: {result.round_number_pct:.0f}%")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _lossaversion_cmd(args: list[str]) -> str:
        """Behavioural bias analyzer: disposition effect and loss aversion.

        Usage: /lossav [LIMIT]  (default 500 trades)
        Compares hold times and PnL% for winners vs. losers.
        Flags if you hold losers longer than winners (disposition effect)
        or if your losses are systematically larger than your wins.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.loss_aversion import compute as la_compute

        result = la_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need both wins and losses with timestamps)."

        disp_icon = "⚠️" if result.disposition_effect else "✅"
        la_icon = {"excessive": "🚨", "high": "⚠️", "normal": "✅", "low": "✅"}.get(result.loss_aversion_level, "")

        lines = [
            f"── Behavioural Analysis ({result.n_trades} trades) ──",
            "",
            f"  Disposition Effect: {disp_icon} {result.disposition_strength.upper()}",
            f"    Hold winner:  {result.median_hold_winner_min:.0f} min (median)",
            f"    Hold loser:   {result.median_hold_loser_min:.0f} min (median)",
            f"    Hold ratio:   {result.hold_ratio:.2f}×  (>1.2 = disposition bias)",
            "",
            f"  Loss Aversion:    {la_icon} {result.loss_aversion_level.upper()}",
            f"    Avg win:  +{result.avg_win_pct:.2f}%",
            f"    Avg loss: -{result.avg_loss_pct:.2f}%",
            f"    Loss mult: {result.loss_multiplier:.2f}×  (<1 = good, >2 = risky)",
            "",
            f"  Premature exits:  {result.premature_exit_rate:.0f}% of winners below avg",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _corrbreakdown_cmd(args: list[str]) -> str:
        """Correlation Breakdown Monitor: detects crisis correlation surges.

        Usage: /corrb  (monitors all open positions)
        Compares pairwise correlations between positions in the first half vs.
        second half of the bar history. Flags correlation surges (all moving
        together = diversification failing) or collapses (assets decorrelating).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "no open positions"
        if len(positions) < 2:
            return "need 2+ open positions for correlation analysis"

        symbols = [p.symbol for p in positions][:8]
        bars_by_symbol: dict[str, list] = {}
        for sym in symbols:
            try:
                bars_by_symbol[sym] = data.get_bars(sym, limit=80)
            except Exception:
                pass

        from amms.analysis.corr_breakdown import analyze as cb_analyze

        result = cb_analyze(bars_by_symbol)
        if result is None:
            return "Not enough data (need 2+ symbols with 10+ bars each)."

        surge_icon = "🚨" if result.corr_surge else ("⚠️" if result.avg_recent_corr > 0.6 else "✅")
        lines = [
            f"── Correlation Breakdown ({', '.join(result.symbols)}) ──",
            f"  Status:  {surge_icon} {'SURGE' if result.corr_surge else 'COLLAPSE' if result.corr_collapse else 'stable'}",
            f"  Baseline avg: {result.avg_baseline_corr:.2f}  →  Recent avg: {result.avg_recent_corr:.2f}",
            "",
        ]
        if result.broken_pairs:
            lines.append(f"  Broken pairs (|Δ|>{result.surge_threshold:.1f}):")
            for p in result.broken_pairs[:5]:
                icon = "↑" if p.delta > 0 else "↓"
                lines.append(f"    {p.sym1}/{p.sym2}:  {p.baseline_corr:+.2f} → {p.recent_corr:+.2f}  ({icon}{abs(p.delta):.2f})")
            lines.append("")
        else:
            lines.append("  No significant pair changes.")
            lines.append("")

        lines.append(result.verdict)
        return "\n".join(lines)

    def _cvar_cmd(args: list[str]) -> str:
        """Expected Shortfall / CVaR: average loss in the worst X% of scenarios.

        Usage: /cvar [SYM]
          (no args): CVaR from historical closed trade PnL distribution
          (SYM): CVaR from bar return distribution for that ticker
        Shows VaR and CVaR at 90%, 95%, 99% confidence levels.
        """
        if conn is None and data is None:
            return "DB and data client not wired."

        from amms.analysis.cvar import from_bars, from_trades

        if args:
            # Bar-based CVaR
            sym = args[0].upper()
            if data is None:
                return "Data client not wired."
            try:
                bars = data.get_bars(sym, limit=120)
            except Exception as e:
                return f"data error: {e!r}"
            result = from_bars(bars, symbol=sym)
            if result is None:
                return f"Need 30+ bars for {sym}."
        else:
            # Trade-based CVaR
            if conn is None:
                return "DB not wired."
            result = from_trades(conn)
            if result is None:
                return "Need 10+ closed trades with losses."

        risk_icon = {"extreme": "🚨", "high": "⚠️", "moderate": "🟡", "low": "✅"}.get(result.tail_risk_label, "")
        lines = [
            f"── CVaR / Expected Shortfall ({result.source}) ──",
            f"  Tail risk:  {risk_icon} {result.tail_risk_label.upper()}",
            f"  Tail/avg:   {result.tail_risk_score:.2f}×",
            "",
            f"  {'Conf':>6}  {'VaR':>8}  {'CVaR':>8}  {'N tail':>7}",
        ]
        for lv in result.levels:
            lines.append(
                f"  {lv.confidence*100:.0f}%    "
                f"{lv.var:>7.2f}%  {lv.cvar:>7.2f}%  {lv.n_tail_obs:>6}"
            )
        lines += [
            "",
            f"  Max single loss:  {result.max_loss:.2f}%",
            f"  Avg loss:         {result.avg_loss:.2f}%",
            f"  N observations:   {result.n_observations}  ({result.n_losses} losses)",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _gapfill_cmd(args: list[str]) -> str:
        """Gap Fill Probability: historical gap analysis for open positions.

        Usage: /gapfill [SYM]  (default: all open positions)
        Analyses historical overnight gaps and estimates fill rate, fill time,
        and size-conditional probabilities. Flags any current open gap.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /gapfill AAPL)"

        from amms.analysis.gap_fill import analyze as gf_analyze

        lines = []
        for sym in symbols[:5]:
            try:
                bars = data.get_bars(sym, limit=120)
            except Exception:
                bars = []
            result = gf_analyze(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 20+ bars)")
                continue

            if len(result.gaps) == 0:
                lines.append(f"  {sym:<6}  no gaps detected ({result.bars_analysed} bars, threshold {result.gap_threshold_pct}%)")
                continue

            lines += [
                f"── {sym} Gap Fill ({result.bars_analysed} bars) ──",
                f"  Gaps:    {len(result.gaps)} total  "
                f"(↑{result.n_up_gaps} up / ↓{result.n_down_gaps} down)",
                f"  Fill:    {result.fill_rate:.0f}%  "
                f"(↑{result.up_fill_rate:.0f}% / ↓{result.down_fill_rate:.0f}%)",
                f"  Avg fill time: {result.avg_bars_to_fill:.1f} bars",
            ]
            if result.small_fill_rate or result.large_fill_rate:
                lines.append(
                    f"  Size:    small {result.small_fill_rate:.0f}% / "
                    f"large {result.large_fill_rate:.0f}%"
                )
            if result.current_gap:
                g = result.current_gap
                lines.append(
                    f"  ⚡ CURRENT GAP: {g.kind} {g.gap_pct:+.2f}%  "
                    f"→ fill prob {result.current_gap_fill_prob:.0f}%"
                )
            lines += [f"  {result.verdict}", ""]

        return "\n".join(lines).rstrip()

    def _exitquality_cmd(args: list[str]) -> str:
        """Trade exit quality: were you exiting at optimal times?

        Usage: /exitq [LIMIT]  (default 500 trades)
        Classifies each trade exit as optimal / too_early / late_exit / normal.
        Flags patterns like cutting winners early or letting losers run.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.exit_quality import compute as eq_compute

        result = eq_compute(conn, limit=limit)
        if result is None:
            return "Need 5+ trades with both wins and losses."

        q_icon = "✅" if result.avg_exit_quality >= 0.6 else ("⚠️" if result.avg_exit_quality >= 0.4 else "🔴")
        lines = [
            f"── Exit Quality ({result.n_trades} trades) ──",
            f"  Avg quality:  {q_icon} {result.avg_exit_quality:.2f}/1.0",
            f"  Consistency:  {result.exit_consistency:.2f}",
            "",
            f"  {'Classification':14}  {'Count':>5}  {'%':>6}",
            f"  {'Optimal':14}  {result.n_winners + result.n_losers - int(result.too_early_pct/100*result.n_trades) - int(result.late_exit_pct/100*result.n_trades):>5}  {result.optimal_pct:>5.0f}%",
            f"  {'Too early':14}  {int(result.too_early_pct/100*result.n_trades):>5}  {result.too_early_pct:>5.0f}%",
            f"  {'Late exit':14}  {int(result.late_exit_pct/100*result.n_trades):>5}  {result.late_exit_pct:>5.0f}%",
            f"  {'Normal':14}  {int(result.normal_pct/100*result.n_trades):>5}  {result.normal_pct:>5.0f}%",
            "",
            f"  Avg winner: +{result.avg_winner_pct:.2f}%  |  Avg loser: {result.avg_loser_pct:.2f}%",
            f"  Best exit: {result.best_exit_pnl:+.2f}%  |  Worst: {result.worst_exit_pnl:+.2f}%",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _calendaranomaly_cmd(args: list[str]) -> str:
        """Calendar Anomaly Detector: day-of-week, month, Q-end, month-start effects.

        Usage: /calanom [LIMIT]  (default 1000 trades)
        Tests if you systematically perform better on specific days or months.
        """
        if conn is None:
            return "DB not wired."

        limit = 1000
        if args:
            try:
                limit = max(10, min(int(args[0]), 5000))
            except ValueError:
                pass

        from amms.analysis.calendar_anomaly import compute as ca_compute

        result = ca_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ closed trades)."

        lines = [f"── Calendar Anomalies ({result.n_trades} trades) ──", ""]

        # Weekday table
        lines.append("  Day-of-Week:")
        for s in result.by_weekday:
            if s.n_trades == 0:
                continue
            marker = "★" if s == result.best_weekday else ("☆" if s == result.worst_weekday else " ")
            rel = "" if s.reliable else " ⚠️"
            lines.append(f"  {marker} {s.label:<10}  {s.n_trades:>4}×  WR {s.win_rate:>4.0f}%  avg {s.avg_pnl_pct:>+6.2f}%{rel}")
        lines.append("")

        # Month table (only months with trades)
        months_with_trades = [s for s in result.by_month if s.n_trades > 0]
        if months_with_trades:
            lines.append("  Monthly (trades > 0):")
            for s in sorted(months_with_trades, key=lambda x: -x.avg_pnl_pct)[:6]:
                marker = "★" if s == result.best_month else ("☆" if s == result.worst_month else " ")
                rel = "" if s.reliable else " ⚠️"
                lines.append(f"  {marker} {s.label:<5}  {s.n_trades:>4}×  WR {s.win_rate:>4.0f}%  avg {s.avg_pnl_pct:>+6.2f}%{rel}")
            lines.append("")

        # Q-end and month-start effects
        qe, qo = result.qend_vs_other
        if qe.n_trades > 0:
            lines.append(f"  Q-end:       {qe.n_trades}× avg {qe.avg_pnl_pct:+.2f}%  vs  Other: {qo.avg_pnl_pct:+.2f}%")
        ms, mo = result.month_start_vs_other
        if ms.n_trades > 0:
            lines.append(f"  Month-start: {ms.n_trades}× avg {ms.avg_pnl_pct:+.2f}%  vs  Other: {mo.avg_pnl_pct:+.2f}%")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _sigstrength_cmd(args: list[str]) -> str:
        """Technical Signal Strength Aggregator: 6-factor bull/bear score.

        Usage: /sigstr SYMBOL [BARS]  (default 120 bars)
        Combines trend, momentum, RSI, BB, volume, EMA slope into 0-100 score.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 120
        for a in args:
            if a.isdigit():
                bar_count = max(60, min(int(a), 500))

        if not parts:
            return "Usage: /sigstr SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.signal_strength import analyze as ss_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = ss_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 55+)."

        bar_visual = "█" * int(result.score // 10) + "░" * (10 - int(result.score // 10))

        lines = [f"── Signal Strength: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Score:  {result.score:.0f}/100  {bar_visual}  [{result.grade.upper()}]")
        lines.append(f"  Vote:   {result.total_vote:+d} / {result.max_possible}")
        lines.append(f"  Price:  {result.current_price:.2f}")
        lines.append("")

        lines.append("  Components:")
        for c in result.components:
            vote_str = f"{'+' if c.vote >= 0 else ''}{c.vote}"
            lines.append(f"    {c.name:<14} {vote_str:>3}  {c.description}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _swingpts_cmd(args: list[str]) -> str:
        """Swing High/Low Detector: pivot points and market structure.

        Usage: /swingpts SYMBOL [BARS]  (default 120 bars)
        Identifies HH/HL/LH/LL structure, nearest support and resistance.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 120
        for a in args:
            if a.isdigit():
                bar_count = max(20, min(int(a), 500))

        if not parts:
            return "Usage: /swingpts SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.swing_points import analyze as sp_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = sp_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 15+)."

        struct_arrow = {"uptrend": "▲", "downtrend": "▼", "sideways": "─", "unknown": "?"}.get(result.structure, "?")
        hh_str = "HH ✓" if result.hh else "HH ✗"
        hl_str = "HL ✓" if result.hl else "HL ✗"
        lh_str = "LH ✓" if result.lh else "LH ✗"
        ll_str = "LL ✓" if result.ll else "LL ✗"

        lines = [f"── Swing Points: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Structure: {struct_arrow} {result.structure.upper()}  |  {hh_str}  {hl_str}  {lh_str}  {ll_str}")
        lines.append(f"  Pivots:    {len(result.swing_highs)} highs, {len(result.swing_lows)} lows")
        lines.append(f"  Price:     {result.current_price:.2f}")

        if result.nearest_resistance is not None:
            lines.append(f"  Resistance:{result.nearest_resistance:.2f}  ({result.resistance_distance_pct:.1f}% above)")
        if result.nearest_support is not None:
            lines.append(f"  Support:   {result.nearest_support:.2f}  ({result.support_distance_pct:.1f}% below)")

        lines.append("")
        if result.recent_high:
            lines.append(f"  Last swing H: bar {result.recent_high.bar_idx} @ {result.recent_high.price:.2f}")
        if result.recent_low:
            lines.append(f"  Last swing L: bar {result.recent_low.bar_idx} @ {result.recent_low.price:.2f}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _pascore_cmd(args: list[str]) -> str:
        """Price Action Composite Score: 6-factor pure price analysis.

        Usage: /pascore SYMBOL [BARS]  (default 150 bars)
        Scores trend, momentum, mean reversion, volatility, bar quality, range expansion.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 150
        for a in args:
            if a.isdigit():
                bar_count = max(110, min(int(a), 500))

        if not parts:
            return "Usage: /pascore SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.price_action_score import analyze as pa_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = pa_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 105+)."

        score_bar = "█" * int(result.composite // 10) + "░" * (10 - int(result.composite // 10))

        lines = [f"── Price Action Score: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Score:  {result.composite:.0f}/100  [{score_bar}]  {result.grade.upper().replace('_', ' ')}")
        lines.append(f"  Price:  {result.current_price:.2f}  ROC20: {result.roc20:+.1f}%")
        lines.append(f"  Z:      {result.z_score:+.2f}  ATR: {result.atr_pct:.1f}%")
        lines.append("")
        lines.append("  Factors:")
        for f in result.factors:
            bar = "█" * int(f.score // 10) + "░" * (10 - int(f.score // 10))
            lines.append(f"    {f.name:<20} {f.score:>5.0f}  [{bar}]  {f.description}")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _psar_cmd(args: list[str]) -> str:
        """Parabolic SAR Analyser: stop-and-reverse trend system.

        Usage: /psar SYMBOL [BARS]  (default 100 bars)
        Shows SAR level, direction, acceleration factor, and flip history.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(10, min(int(a), 500))

        if not parts:
            return "Usage: /psar SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.parabolic_sar import analyze as psar_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = psar_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 5+)."

        dir_arrow = "▲" if result.direction == "bull" else "▼"
        dir_label = "BULLISH" if result.direction == "bull" else "BEARISH"

        lines = [f"── Parabolic SAR: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Direction: {dir_arrow} {dir_label}")
        lines.append(f"  Price:     {result.current_price:.2f}")
        lines.append(f"  SAR:       {result.sar:.2f}  ({result.distance_pct:.1f}% away)")
        lines.append(f"  AF:        {result.current_af:.3f}  (max: {result.af_max})")
        lines.append(f"  EP:        {result.current_ep:.2f}")
        lines.append(f"  Age:       {result.trend_age} bars  |  Flips: {result.flip_count}")
        lines.append(f"  Bull:      {result.bull_pct:.0f}%  |  Avg duration: {result.avg_trend_duration:.1f} bars")

        if result.history:
            recent_flips = [s for s in result.history if s.flipped]
            if recent_flips:
                lines.append("")
                lines.append("  Recent flips:")
                for s in recent_flips[-3:]:
                    lines.append(f"    bar {s.bar_idx}: → {s.direction.upper()} @ {s.price:.2f}  SAR={s.sar:.2f}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _cmf_cmd(args: list[str]) -> str:
        """Chaikin Money Flow: volume-weighted buying/selling pressure.

        Usage: /cmf SYMBOL [BARS]  (default 80 bars)
        Shows CMF value (-1 to +1), accumulation/distribution signal, trend.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 80
        for a in args:
            if a.isdigit():
                bar_count = max(25, min(int(a), 500))

        if not parts:
            return "Usage: /cmf SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.chaikin_mf import analyze as cmf_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = cmf_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 22+)."

        cmf_pct = int((result.cmf + 1.0) / 2.0 * 10)
        cmf_bar = "░" * cmf_pct + "▓" + "░" * (10 - cmf_pct)

        sig_arrow = {"strong_buy": "▲▲", "buy": "▲", "neutral": "─",
                     "sell": "▼", "strong_sell": "▼▼"}.get(result.signal, "─")

        lines = [f"── Chaikin Money Flow: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  CMF:     {result.cmf:+.3f}  [{cmf_bar}]")
        lines.append(f"  Signal:  {sig_arrow} {result.signal.upper().replace('_', ' ')}")
        lines.append(f"  Trend:   {result.cmf_trend.upper()}  ({result.trend_bars} bars)")
        lines.append(f"  +Flow:   {result.above_zero_pct:.0f}% of bars positive")
        lines.append(f"  Crosses: {result.zero_crossings}  |  Avg Vol: {result.avg_volume:,.0f}")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _donchian_cmd(args: list[str]) -> str:
        """Donchian Channel Analyser: rolling high/low breakout channel.

        Usage: /donchian SYMBOL [BARS]  (default 80 bars)
        Shows N-bar high/low channel, breakout status, and channel width trend.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 80
        for a in args:
            if a.isdigit():
                bar_count = max(25, min(int(a), 500))

        if not parts:
            return "Usage: /donchian SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.donchian_channel import analyze as dc_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = dc_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 22+)."

        pos_pct = int(result.price_position * 10)
        channel_bar = "░" * pos_pct + "▓" + "░" * (10 - pos_pct)

        bo_tag = ""
        if result.breakout_up:
            bo_tag = "  ▲ NEW HIGH"
        elif result.breakout_down:
            bo_tag = "  ▼ NEW LOW"
        elif result.near_upper:
            bo_tag = "  ~ near upper"
        elif result.near_lower:
            bo_tag = "  ~ near lower"

        lines = [f"── Donchian Channel: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Price:  {result.current_price:.2f}{bo_tag}")
        lines.append(f"  Upper:  {result.upper:.2f}  ({result.period}-bar high)")
        lines.append(f"  Middle: {result.middle:.2f}")
        lines.append(f"  Lower:  {result.lower:.2f}  ({result.period}-bar low)")
        lines.append(f"  Width:  {result.channel_width_pct:.1f}%  [{channel_bar}]  {result.channel_trend}")
        lines.append(f"  Since upper: {result.bars_since_upper} bars  |  Since lower: {result.bars_since_lower} bars")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _supertrend_cmd(args: list[str]) -> str:
        """Supertrend Indicator: ATR-based trend-following with flip detection.

        Usage: /supertrend SYMBOL [BARS]  (default 100 bars)
        Single bull/bear signal with dynamic support/resistance level.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(20, min(int(a), 500))

        if not parts:
            return "Usage: /supertrend SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.supertrend import analyze as st_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = st_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 15+)."

        dir_arrow = "▲" if result.direction == "bull" else "▼"
        dir_label = "BULLISH" if result.direction == "bull" else "BEARISH"

        lines = [f"── Supertrend: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Direction:  {dir_arrow} {dir_label}")
        lines.append(f"  Price:      {result.current_price:.2f}")
        lines.append(f"  ST Level:   {result.supertrend_level:.2f}  ({result.distance_pct:+.1f}%)")
        lines.append(f"  ATR:        {result.atr:.2f}")
        lines.append(f"  Trend age:  {result.trend_age} bars")
        lines.append(f"  Flips (history): {result.flip_count}")

        if result.history:
            lines.append("")
            lines.append("  Recent flips:")
            flips = [s for s in result.history if s.flipped]
            if flips:
                for s in flips[-3:]:
                    lines.append(f"    bar {s.bar_idx}: → {s.direction.upper()} @ {s.price:.2f}")
            else:
                lines.append("    None in lookback window")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _keltner_cmd(args: list[str]) -> str:
        """Keltner Channel Analyser: EMA + ATR bands for trend & breakout.

        Usage: /keltner SYMBOL [BARS]  (default 100 bars)
        Shows EMA middle with ±2 ATR upper/lower bands and price position.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(30, min(int(a), 500))

        if not parts:
            return "Usage: /keltner SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.keltner_channel import analyze as kc_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = kc_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 30+)."

        pos_pct = int(min(result.price_position, 1.0) * 10)
        channel_bar = "░" * pos_pct + "▓" + "░" * (10 - pos_pct)

        bo_tag = ""
        if result.breakout_up:
            bo_tag = "  ▲ BREAKOUT UP"
        elif result.breakout_down:
            bo_tag = "  ▼ BREAKOUT DOWN"

        lines = [f"── Keltner Channel: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Price:  {result.current_price:.2f}{bo_tag}")
        lines.append(f"  Upper:  {result.upper:.2f}")
        lines.append(f"  Middle: {result.middle:.2f}  (EMA-{result.period})")
        lines.append(f"  Lower:  {result.lower:.2f}")
        lines.append(f"  Width:  {result.channel_width_pct:.1f}%  [{channel_bar}]  {result.position_label}")
        lines.append(f"  Trend:  {result.trend_direction.upper()}  ({result.trend_bars} bars)")
        lines.append(f"  ATR:    {result.atr:.2f} ({result.atr_pct:.1f}%)")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _candle_cmd(args: list[str]) -> str:
        """Candlestick Pattern Recognizer: detects hammer, engulfing, stars, etc.

        Usage: /candle SYMBOL [BARS]  (default 60 bars scanned)
        Identifies single-, two-, and three-bar reversal and continuation patterns.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 60
        for a in args:
            if a.isdigit():
                bar_count = max(5, min(int(a), 500))

        if not parts:
            return "Usage: /candle SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.candlestick_patterns import analyze as candle_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = candle_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 3+)."

        bias_arrow = {"bullish": "▲", "bearish": "▼", "neutral": "─"}.get(result.dominant_bias, "─")
        score_bar = "█" * int(abs(result.bias_score) * 10) + "░" * (10 - int(abs(result.bias_score) * 10))

        lines = [f"── Candlestick Patterns: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Dominant Bias:  {bias_arrow} {result.dominant_bias.upper()}  [{score_bar}]  score {result.bias_score:+.2f}")
        lines.append(f"  Patterns found: {len(result.patterns)}  (bull: {result.bullish_count}, bear: {result.bearish_count})")
        lines.append("")

        if result.recent_patterns:
            lines.append("  Recent patterns (last 5 bars):")
            for p in result.recent_patterns[-5:]:
                arrow = "▲" if p.bias == "bullish" else ("▼" if p.bias == "bearish" else "─")
                lines.append(f"    {arrow} {p.name:<26} [{p.strength}]  {p.description}")
        else:
            lines.append("  No patterns in last 5 bars.")

        if result.last_signal:
            lines.append("")
            lines.append(f"  Last signal: {result.last_signal.name} @ bar {result.last_signal.bar_idx}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _mrband_cmd(args: list[str]) -> str:
        """Mean Reversion Band Analyser: z-score distance from rolling mean.

        Usage: /mrband SYMBOL [BARS]  (default 80 bars)
        Shows -2σ/-1σ/mean/+1σ/+2σ bands and reversion signal.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 80
        for a in args:
            if a.isdigit():
                bar_count = max(32, min(int(a), 500))

        if not parts:
            return "Usage: /mrband SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.mean_reversion_band import analyze as mrb_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = mrb_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 30+)."

        bar_z = max(0, min(10, int((result.z_score + 2) / 4 * 10)))
        visual = "░" * bar_z + "▓" + "░" * (10 - bar_z)

        lines = [f"── Mean Reversion Bands: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Price:  {result.current_price:.2f}  ({result.pct_from_mean:+.1f}% from mean)")
        lines.append(f"  Z-score: {result.z_score:+.2f}  [{visual}]  {result.signal.upper()}")
        lines.append(f"  Percentile: {result.z_percentile:.0f}th  |  Strength: {result.signal_strength:.2f}")
        if result.half_life is not None:
            lines.append(f"  Half-life: {result.half_life:.1f} bars")
        lines.append("")
        lines.append("  Bands:")
        for b in reversed(result.bands):
            marker = " ◄" if abs(result.current_price - b.price) < result.std * 0.3 else ""
            lines.append(f"    {b.sigma:+.0f}σ  {b.price:.2f}{marker}")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _regtrans_cmd(args: list[str]) -> str:
        """Regime Transition Detector: bull/neutral/bear state changes.

        Usage: /regtrans SYMBOL [BARS]  (default 150 bars)
        Multi-factor vote: SMA-50/200, ROC-20, RSI, ATR.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 150
        for a in args:
            if a.isdigit():
                bar_count = max(60, min(int(a), 500))

        if not parts:
            return "Usage: /regtrans SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.regime_transition import analyze as rt_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = rt_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 55+)."

        lines = [f"── Regime Transition: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Current regime:   {result.current_regime.upper()}  (vote {result.current_vote:+d}/4)")
        lines.append(f"  Previous regime:  {result.previous_regime}")
        if result.transition_detected:
            lines.append(f"  TRANSITION: {result.transition_direction}")
        else:
            lines.append(f"  No transition  ({result.transition_direction})")
        if result.transition_bars_ago is not None:
            lines.append(f"  Last change:  {result.transition_bars_ago} bars ago")
        lines.append("")
        lines.append(f"  Price: {result.current_price:.2f}  RSI: {result.rsi:.0f}  ROC-20: {result.roc20:+.1f}%")
        if result.sma50:
            lines.append(f"  SMA-50:  {result.sma50:.2f}  ({'+' if result.current_price > result.sma50 else '-'}above)")
        if result.sma200:
            lines.append(f"  SMA-200: {result.sma200:.2f}")
        lines.append(f"  ATR: {result.atr_pct:.2f}%  (pct {result.atr_percentile:.0f}%)")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _volfcast_cmd(args: list[str]) -> str:
        """EWMA Volatility Forecast: expected daily vol and 1-day 95% VaR.

        Usage: /volfcast SYMBOL [BARS]  (default 100 bars)
        Uses RiskMetrics EWMA model (λ=0.94) for volatility forecast.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(20, min(int(a), 500))

        if not parts:
            return "Usage: /volfcast SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.volatility_forecast import analyze as vf_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = vf_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 15+)."

        lines = [f"── EWMA Vol Forecast: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Current daily vol:   {result.ewma_vol_daily:.3f}%")
        lines.append(f"  Annualised vol:      {result.ewma_vol_annual:.1f}%")
        lines.append(f"  1-day 95% VaR:       {result.var_95_1d:.2f}%")
        lines.append(f"  Vol percentile:      {result.vol_percentile:.0f}%  ({result.vol_trend})")
        lines.append("")
        lines.append(f"  Forward forecasts (daily vol):")
        lines.append(f"    1-day:  {result.vol_1d:.3f}%")
        lines.append(f"    5-day:  {result.vol_5d:.3f}%")
        lines.append(f"    10-day: {result.vol_10d:.3f}%")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _exhaust_cmd(args: list[str]) -> str:
        """Trend Exhaustion Detector: price extension, RSI divergence, momentum decay.

        Usage: /exhaust SYMBOL [BARS]  (default 100 bars)
        Score 0-100: higher = trend more likely to reverse.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(60, min(int(a), 500))

        if not parts:
            return "Usage: /exhaust SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.trend_exhaustion import analyze as ex_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = ex_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 55+)."

        score_bar = "█" * int(result.exhaustion_score // 10) + "░" * (10 - int(result.exhaustion_score // 10))

        lines = [f"── Trend Exhaustion: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Exhaustion: {result.exhaustion_score:.0f}/100  {score_bar}  [{result.exhaustion_label.upper()}]")
        lines.append(f"  Trend:      {result.trend_direction}-trend  |  Price: {result.current_price:.2f}")
        lines.append(f"  RSI:        {result.rsi:.1f}  ROC-5: {result.roc5:+.2f}%  ROC-20: {result.roc20:+.2f}%")
        if result.sma20:
            lines.append(f"  SMA-20:     {result.sma20:.2f}  ({result.pct_above_sma20:+.1f}% away)")
        lines.append("")
        lines.append("  Signals:")
        lines.append(f"    Price extension:   {result.price_extension * 100:.0f}%")
        lines.append(f"    RSI divergence:    {'YES' if result.rsi_divergence else 'no'}")
        lines.append(f"    Momentum decay:    {'YES' if result.momentum_decay else 'no'}")
        lines.append(f"    ATR contraction:   {'YES' if result.atr_contraction else 'no'}  (ATR {result.atr_pct:.2f}%)")
        lines.append(f"    Wick rejection:    {result.bar_rejection * 100:.0f}%")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _fib_cmd(args: list[str]) -> str:
        """Fibonacci Retracement Levels: swing high/low Fib levels and nearest support/resistance.

        Usage: /fib SYMBOL [WINDOW]  (default swing window = 50 bars, loads 200 bars)
        Shows all Fib levels and where price currently sits.
        """
        if data is None:
            return "Data client not wired."

        parts = [a for a in args if not a.isdigit()]
        swing_window = 50
        for a in args:
            if a.isdigit():
                swing_window = max(10, min(int(a), 200))

        if not parts:
            return "Usage: /fib SYMBOL [WINDOW]"

        symbol = parts[0].upper()
        bar_count = swing_window + 100

        from amms.analysis.fib_retracement import analyze as fib_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = fib_analyze(bars, symbol=symbol, swing_window=swing_window)
        if result is None:
            return f"Not enough bars or no valid swing found for {symbol}."

        lines = [f"── Fibonacci Levels: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Swing: {result.swing_low:.2f} → {result.swing_high:.2f}  ({result.trend_direction}-trend)")
        lines.append(f"  Price: {result.current_price:.2f}  (retracement depth {result.retracement_depth:.0f}%)")
        lines.append("")

        lines.append("  Levels:")
        for l in result.levels:
            marker = " ◄" if l == result.nearest_level else ""
            sup = " [S]" if l == result.nearest_support else ""
            res = " [R]" if l == result.nearest_resistance else ""
            near = " ~near~" if l.is_near else ""
            lines.append(
                f"    {l.label:>7}  {l.price:>10.2f}  ({l.pct_from_current:>+7.2f}%){near}{marker}{sup}{res}"
            )

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _breadthproxy_cmd(args: list[str]) -> str:
        """Market Breadth Proxy: % of symbols above SMA-50, RSI, ROC-20.

        Usage: /breadth SYM1 SYM2 ... [BARS]  (default 100 bars)
        Scores market-wide participation: broad_bull/neutral/mixed/broad_bear.
        """
        if data is None:
            return "Data source not wired."

        symbols = [a.upper() for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(60, min(int(a), 500))

        if not symbols:
            return "Usage: /breadth SYM1 SYM2 ...  — provide at least 2 symbols."

        from amms.analysis.breadth_proxy import analyze as bp_analyze

        bars_by_sym: dict[str, list] = {}
        for sym in symbols:
            try:
                b = data.get_bars(sym, limit=bar_count)
                if b:
                    bars_by_sym[sym] = b
            except Exception:
                pass

        result = bp_analyze(bars_by_sym)
        if result is None:
            return "Not enough data (need 2+ symbols with 25+ bars each)."

        label_icons = {"broad_bull": "🟢", "neutral": "🟡", "mixed": "🟠", "broad_bear": "🔴"}
        icon = label_icons.get(result.breadth_label, " ")

        lines = [f"── Market Breadth ({result.n_evaluated}/{result.n_symbols} evaluated) ──", ""]
        lines.append(f"  Score: {result.breadth_score:.0f}/100  {icon} {result.breadth_label.upper()}")
        lines.append(f"  Above SMA-50:     {result.pct_above_sma50:.0f}%  ({result.advance_count}/{result.n_evaluated})")
        lines.append(f"  Positive ROC-20:  {result.pct_positive_roc20:.0f}%")
        lines.append(f"  RSI > 50:         {result.pct_rsi_above_50:.0f}%")
        lines.append(f"  Near highs:       {result.pct_near_highs:.0f}%")
        lines.append(f"  Near lows:        {result.pct_near_lows:.0f}%")
        if result.breadth_thrust:
            lines.append(f"  ⚡ BREADTH THRUST: {result.thrust_direction.upper()}")
        lines.append("")

        lines.append("  Symbol detail:")
        for s in sorted(result.symbols, key=lambda x: -x.roc_20):
            flags = []
            if s.above_sma50:
                flags.append("SMA+")
            if s.roc20_positive:
                flags.append("ROC+")
            if s.rsi_above_50:
                flags.append("RSI+")
            if s.near_52w_high:
                flags.append("HIGH")
            if s.near_52w_low:
                flags.append("low")
            lines.append(f"    {s.symbol:<8}  {s.current_price:>8.2f}  ROC {s.roc_20:>+6.1f}%  {' '.join(flags)}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _bbsqueeze_cmd(args: list[str]) -> str:
        """Bollinger Band Squeeze Detector: compressed volatility before breakout.

        Usage: /bbsq SYMBOL [BARS]  (default 150 bars)
        Detects when BB width drops to local minimum (squeeze) and shows bias direction.
        """
        if data is None:
            return "Data source not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 150
        for a in args:
            if a.isdigit():
                bar_count = max(80, min(int(a), 500))

        if not parts:
            return "Usage: /bbsq SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.bb_squeeze import analyze as bb_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = bb_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 75+)."

        sq_bar = "█" * int(result.current_squeeze_score // 10) + "░" * (10 - int(result.current_squeeze_score // 10))

        lines = [f"── BB Squeeze: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Squeeze score:  {result.current_squeeze_score:.0f}/100  {sq_bar}")
        lines.append(f"  BW percentile:  {result.bandwidth_percentile:.0f}%  (0=tightest, 100=widest)")
        lines.append(f"  Squeezed:       {'YES ⚠' if result.is_squeezed else 'no'}  ({result.squeeze_active_bars} bars active)")
        lines.append(f"  Direction bias: {result.direction_bias}")
        lines.append(f"  Price:          {result.current_price:.2f}  (position in band: {result.price_position * 100:.0f}%)")
        lines.append(f"  BB:             {result.lower:.2f} – {result.middle:.2f} – {result.upper:.2f}")
        if result.is_squeezed and result.roc_since_squeeze != 0:
            lines.append(f"  Drift since squeeze start: {result.roc_since_squeeze:+.2f}%")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _mtfmom_cmd(args: list[str]) -> str:
        """Multi-Timeframe Momentum: 5/20/50/100-bar alignment and regime.

        Usage: /mtfmom SYMBOL [BARS]  (default 150 bars)
        Shows momentum direction on 4 timeframes and overall regime.
        """
        if data is None:
            return "Data source not wired."

        parts = [a for a in args if not a.isdigit()]
        bar_count = 150
        for a in args:
            if a.isdigit():
                bar_count = max(105, min(int(a), 500))

        if not parts:
            return "Usage: /mtfmom SYMBOL [BARS]"

        symbol = parts[0].upper()

        from amms.analysis.mtf_momentum import analyze as mtf_analyze

        try:
            bars = data.get_bars(symbol, limit=bar_count)
        except Exception:
            return f"Could not fetch bars for {symbol}."

        if not bars:
            return f"No bar data for {symbol}."

        result = mtf_analyze(bars, symbol=symbol)
        if result is None:
            return f"Not enough bars for {symbol} (need 102+)."

        lines = [f"── MTF Momentum: {result.symbol} ({result.bars_used} bars) ──", ""]
        lines.append(f"  Regime: {result.regime}  (score {result.total_score:+d}/4)")
        lines.append(f"  Aligned: {'yes' if result.aligned else 'no'}  |  Divergence: {'yes' if result.divergence else 'no'}")
        lines.append(f"  Price: {result.current_price:.2f}")
        lines.append("")

        header = f"  {'TF':<8} {'ROC':>8} {'RSI':>7} {'EMA slope':>12} {'dir':<10} {'score':>6}"
        lines.append(header)
        lines.append("  " + "-" * 52)
        for w in result.windows:
            lines.append(
                f"  {w.bars:<8} {w.roc:>+7.2f}% {w.rsi:>6.1f}  {w.ema_slope_pct:>+10.4f}%  {w.direction:<10} {w.score:>+4}"
            )

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _pfdecomp_cmd(args: list[str]) -> str:
        """Profit Factor Decomposition: PF, payoff ratio, edge, Kelly.

        Usage: /pfdecomp [LIMIT]  (default 500 trades)
        Shows profit factor, breakeven win rate, edge, and rolling trend.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.profit_factor import compute as pf_compute

        result = pf_compute(conn, limit=limit)
        if result is None:
            return "Not enough data (need 10+ closed trades with both wins and losses)."

        pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else ">100"
        lines = [f"── Profit Factor Decomposition ({result.n_trades} trades) ──", ""]

        lines.append(f"  Profit factor:      {pf_str}")
        lines.append(f"  Win rate:           {result.win_rate:.1f}%  (breakeven {result.breakeven_win_rate:.1f}%,  edge {result.edge:+.1f}pp)")
        lines.append(f"  Payoff ratio:       {result.payoff_ratio:.2f}×  (avg win {result.avg_winner_pct:+.2f}%  /  avg loss {result.avg_loser_pct:+.2f}%)")
        lines.append(f"  Expectancy/trade:   {result.expectancy:+.3f}%")
        lines.append(f"  Gross wins:         {result.gross_wins:+.2f}%  (across {result.n_winners} trades)")
        lines.append(f"  Gross losses:      -{result.gross_losses:.2f}%  (across {result.n_losers} trades)")
        lines.append(f"  Full Kelly:         {result.kelly_pct:.1f}%  (use half = {result.kelly_pct / 2:.1f}%)")
        lines.append("")

        if result.rolling_20:
            lines.append(f"  Rolling PF (20t) trend: {result.rolling_pf_trend}")
            if result.rolling_20.values:
                latest = result.rolling_20.values[-1]
                lines.append(f"  Latest rolling PF: {latest:.2f}")
            lines.append("")

        if result.by_symbol:
            lines.append("  By symbol (top 5):")
            for s in result.by_symbol:
                pf_s = f"{s.profit_factor:.2f}" if s.profit_factor < 100 else ">100"
                lines.append(f"    {s.symbol:<8}  {s.n_trades:>3}×  PF {pf_s}  WR {s.win_rate:.0f}%  payoff {s.payoff_ratio:.2f}×")
            lines.append("")

        lines.append(result.verdict)
        return "\n".join(lines)

    def _tradesector_cmd(args: list[str]) -> str:
        """Trade Sector Performance: which sectors win most in your journal?

        Usage: /tsector [LIMIT]  (default 1000 trades)
        Detects if sector leadership is rotating recently vs historically.
        """
        if conn is None:
            return "DB not wired."

        limit = 1000
        if args:
            try:
                limit = max(10, min(int(args[0]), 5000))
            except ValueError:
                pass

        from amms.analysis.trade_sector import compute as ts_compute

        result = ts_compute(conn, limit=limit)
        if result is None:
            return "Not enough data (need 10+ trades across 2+ sectors)."

        lines = [f"── Sector Performance ({result.n_trades} trades, {result.n_sectors} sectors) ──", ""]

        for s in result.sectors:
            rel = "" if s.is_reliable else " ⚠"
            rot = f"  Δ{s.rotation_score:+.2f}" if abs(s.rotation_score) > 0.1 else ""
            leader_mark = " ★" if s == result.leader else ""
            lines.append(
                f"  {s.sector:<14}  {s.n_trades:>4}×  WR {s.win_rate:>4.0f}%  "
                f"avg {s.avg_pnl_pct:>+6.2f}%  recent {s.recent_avg_pnl:>+6.2f}%{rot}{leader_mark}{rel}"
            )

        lines.append("")
        if result.rotation_detected and result.historical_leader:
            lines.append(f"  ROTATION: was {result.historical_leader.sector}, now {result.leader.sector if result.leader else '?'}")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _holdtime_cmd(args: list[str]) -> str:
        """Position Hold Time Analyser: median hold, best bucket, hold-PnL correlation.

        Usage: /holdtime [LIMIT]  (default 500 trades)
        Breaks down hold times into scalp/intraday/swing/multi-day buckets.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(5, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.hold_time import compute as ht_compute

        result = ht_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history or duration data (need 5+ trades with timestamps)."

        lines = [f"── Hold Time Analysis ({result.n_trades} trades, {result.n_with_duration} with duration) ──", ""]

        lines.append(f"  Median hold:  {result.median_hold_min:.0f} min  ({result.median_hold_min / 60:.1f} h)")
        lines.append(f"  Mean hold:    {result.mean_hold_min:.0f} min")
        lines.append(f"  Range:        {result.min_hold_min:.0f}–{result.max_hold_min:.0f} min")
        lines.append(f"  Hold/PnL corr: {result.hold_pnl_correlation:+.3f}")
        lines.append("")

        lines.append("  Bucket breakdown:")
        for b in result.buckets:
            if b.n_trades == 0:
                continue
            best = " ★" if b == result.best_bucket else ""
            worst = " ☆" if b == result.worst_bucket else ""
            lines.append(f"    {b.label:<12} {b.n_trades:>4}×  WR {b.win_rate:>4.0f}%  avg {b.avg_pnl_pct:>+6.2f}%  ~{b.avg_hold_min:.0f}min{best}{worst}")
        lines.append("")

        if result.by_symbol:
            lines.append("  Top symbols by trade count:")
            for s in result.by_symbol:
                lines.append(f"    {s.symbol:<8}  {s.n_trades:>3}×  avg {s.avg_hold_min:.0f}min  {s.avg_pnl_pct:>+5.2f}%")
            lines.append("")

        lines.append(f"  Hold trend: {result.hold_trend}")
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _streak_cmd(args: list[str]) -> str:
        """Streak Analyser: win/loss streaks, hot-hand effect, conditional probabilities.

        Usage: /streak [LIMIT]  (default 500 trades)
        Shows longest streaks, current streak, and whether wins cluster.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.streak_analyser import compute as streak_compute

        result = streak_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ closed trades)."

        lines = [f"── Streak Analysis ({result.n_trades} trades) ──", ""]

        lines.append(f"  Baseline win rate: {result.baseline_win_rate * 100:.1f}%")
        lines.append("")

        if result.longest_win_streak:
            lw = result.longest_win_streak
            lines.append(f"  Longest WIN streak:  {lw.length}× ({lw.total_pnl:+.1f}% total)")
        if result.longest_loss_streak:
            ll = result.longest_loss_streak
            lines.append(f"  Longest LOSS streak: {ll.length}× ({ll.total_pnl:+.1f}% total)")
        lines.append(f"  Avg win streak len:  {result.avg_win_streak_len:.1f}")
        lines.append(f"  Avg loss streak len: {result.avg_loss_streak_len:.1f}")
        lines.append("")

        if result.p_win_after_win is not None:
            lines.append(f"  P(win | prev win):   {result.p_win_after_win * 100:.1f}%  "
                         f"(baseline {result.baseline_win_rate * 100:.1f}%)")
        if result.hot_hand_effect is not None:
            direction = "hot-hand" if result.hot_hand_effect > 0 else "mean-reversion"
            lines.append(f"  Hot-hand effect:     {result.hot_hand_effect:+.3f} ({direction})")
        if result.p_loss_after_loss is not None:
            lines.append(f"  P(loss | prev loss): {result.p_loss_after_loss * 100:.1f}%")
        if result.cold_hand_effect is not None:
            lines.append(f"  Cold-hand effect:    {result.cold_hand_effect:+.3f}")
        lines.append("")

        if result.current_streak:
            cs = result.current_streak
            lines.append(f"  Current streak: {cs.length}× {cs.kind.upper()} ({cs.total_pnl:+.1f}%)")

        if result.avg_pnl_after_win_streak is not None:
            lines.append(f"  Avg PnL after win streak ends:  {result.avg_pnl_after_win_streak:+.2f}%")
        if result.avg_pnl_after_loss_streak is not None:
            lines.append(f"  Avg PnL after loss streak ends: {result.avg_pnl_after_loss_streak:+.2f}%")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _ddcurve_cmd(args: list[str]) -> str:
        """Drawdown Curve Analyser: equity curve, episodes, ulcer index.

        Usage: /ddcurve [LIMIT]  (default 500 trades)
        Identifies all drawdown periods: depth, duration, recovery time.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(5, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.drawdown_curve import compute as dd_compute

        result = dd_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ closed trades)."

        lines = [f"── Drawdown Curve ({result.n_trades} trades, {result.n_episodes} episodes) ──", ""]

        lines.append(f"  Max drawdown:   {result.max_drawdown_pct:>+7.2f}%  over {result.max_drawdown_duration} trades")
        lines.append(f"  Avg drawdown:   {result.avg_drawdown_pct:>+7.2f}%  avg dur {result.avg_drawdown_duration:.1f} trades")
        lines.append(f"  Current DD:     {result.current_drawdown_pct:>+7.2f}%  {'⚠ IN DRAWDOWN' if result.in_drawdown else '✓ at high'}")
        lines.append(f"  Ulcer Index:    {result.ulcer_index:.4f}")
        if result.max_drawdown_pct < 0:
            lines.append(f"  Recovery factor: {result.recovery_factor:.2f}")
        if result.longest_recovery_bars is not None:
            lines.append(f"  Longest recovery: {result.longest_recovery_bars} trades")
        lines.append("")

        if result.episodes:
            lines.append("  Worst 3 episodes:")
            worst = sorted(result.episodes, key=lambda e: e.depth_pct)[:3]
            for ep in worst:
                rec = f"{ep.recovery_bars}t" if ep.recovery_bars is not None else "ongoing"
                lines.append(f"    depth {ep.depth_pct:>+6.2f}%  dur {ep.duration_bars}t  rec {rec}")
            lines.append("")

        lines.append(result.verdict)
        return "\n".join(lines)

    def _wscore_cmd(args: list[str]) -> str:
        """Watchlist Opportunity Scorer: rank symbols 0-100 by composite score.

        Usage: /wscore SYM1 SYM2 ... [BARS=60]
        Scores: momentum ROC-20 (20%), RSI (20%), volume trend (15%),
                SMA-50 proximity (15%), ATR volatility (15%), EMA trend (15%).
        Grade: A≥80, B≥65, C≥50, D≥35, F<35.
        """
        if data is None:
            return "Data source not wired."

        symbols = [a.upper() for a in args if not a.isdigit()]
        bar_count = 100
        for a in args:
            if a.isdigit():
                bar_count = max(30, min(int(a), 500))

        if not symbols:
            return "Usage: /wscore SYM1 SYM2 ... [BARS]  — provide at least one symbol."

        from amms.analysis.watchlist_scorer import score_many

        bars_by_symbol: dict[str, list] = {}
        for sym in symbols:
            try:
                bars = data.get_bars(sym, limit=bar_count)
                if bars:
                    bars_by_symbol[sym] = bars
            except Exception:
                pass

        if not bars_by_symbol:
            return "Could not fetch bars for any of the requested symbols."

        report = score_many(bars_by_symbol)

        if report.n_graded == 0:
            return "No symbols had sufficient bar history (need 25+ bars)."

        lines = [f"── Watchlist Scorer ({report.n_graded}/{report.n_symbols} graded) ──", ""]
        for s in report.scores:
            bar = "█" * int(s.total_score // 10) + "░" * (10 - int(s.total_score // 10))
            lines.append(f"  {s.grade}  {s.symbol:<6}  {s.total_score:>5.1f}  {bar}")
            lines.append(f"       mom {s.momentum_score:.0f}  rsi {s.rsi_score:.0f}  vol {s.volume_score:.0f}"
                         f"  sma {s.sma_score:.0f}  atr {s.vol_score:.0f}  trnd {s.trend_score:.0f}")
            lines.append(f"       price {s.current_price:.2f}  RSI {s.rsi:.0f}  ROC {s.roc_20:+.1f}%")
            lines.append("")

        if report.top_pick:
            tp = report.top_pick
            lines.append(f"Top pick: {tp.symbol} ({tp.grade}, {tp.total_score:.0f}/100)")
            lines.append(tp.summary)

        return "\n".join(lines)

    def _autocorr_cmd(args: list[str]) -> str:
        """Trade outcome autocorrelation: hot-hand or mean-reversion?

        Usage: /autocorr [LIMIT]  (default 200 trades)
        Tests if wins/losses cluster (hot hand) or alternate (mean-reversion).
        Uses lag-1/2/3 autocorrelation and the runs test.
        """
        if conn is None:
            return "DB not wired."

        limit = 200
        if args:
            try:
                limit = max(20, min(int(args[0]), 1000))
            except ValueError:
                pass

        from amms.analysis.outcome_autocorr import compute as ac_compute

        result = ac_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 20+ trades)."

        interp_icon = {
            "hot_hand": "🔥", "mean_reversion": "↔️ ", "random": "🎲"
        }.get(result.interpretation, "")

        lines = [
            f"── Outcome Autocorrelation ({result.n_trades} trades) ──",
            f"  Pattern:  {interp_icon} {result.interpretation.replace('_', ' ').upper()}",
            f"  Win rate: {result.win_rate:.1f}%",
            "",
            f"  Lag-1 autocorr: {result.lag1_autocorr:+.3f}",
            f"  Lag-2 autocorr: {result.lag2_autocorr:+.3f}",
            f"  Lag-3 autocorr: {result.lag3_autocorr:+.3f}",
            "",
            f"  Runs test: {result.runs_count} runs (expected {result.expected_runs:.0f})",
            f"  Z-score: {result.runs_z_score:+.2f}  "
            f"{'← significant (p<0.05)' if result.runs_significant else '← not significant'}",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _sectorwr_cmd(args: list[str]) -> str:
        """Win rate and PnL by sector from closed trade history.

        Usage: /sectorwr [LIMIT]  (default 500 trades)
        Maps traded symbols to GICS sectors and shows win rate / PnL per sector.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(5, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.sector_win_rate import compute as swr_compute

        result = swr_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ trades)."

        lines = [
            f"── Sector Win Rate ({result.n_trades} trades, {result.n_sectors} sectors) ──",
            "",
            f"  {'Sector':<22} {'Trades':>6} {'WR':>6} {'Avg%':>8} {'Total PnL':>10}",
            f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*10}",
        ]
        for s in result.sectors:
            mark = " ← best" if s.sector == result.best_sector else (" ← worst" if s.sector == result.worst_sector else "")
            lines.append(
                f"  {s.sector:<22} {s.n_trades:>6} "
                f"{s.win_rate:>5.0f}% {s.avg_pnl_pct:>+8.2f}% {s.total_pnl:>+10.2f}{mark}"
            )

        if result.unknown_trades > 0:
            lines.append(f"\n  ({result.unknown_trades} trades could not be mapped to a sector)")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _wrstab_cmd(args: list[str]) -> str:
        """Win rate stability: how consistent is the win rate over time?

        Usage: /wrstab [WINDOW]  (default 20 trades per window)
        Splits trade history into rolling windows and measures win rate
        variability. High CV = luck-dependent; low CV = consistent edge.
        """
        if conn is None:
            return "DB not wired."

        window_size = 20
        if args:
            try:
                window_size = max(10, min(int(args[0]), 50))
            except ValueError:
                pass

        from amms.analysis.win_rate_stability import compute as wrs_compute

        result = wrs_compute(conn, window_size=window_size)
        if result is None:
            return f"Not enough trade history (need {window_size * 2}+ trades)."

        grade_icon = {
            "Excellent": "🏆", "Good": "✅", "Moderate": "⚠️", "Unstable": "🔴"
        }.get(result.stability_grade, "")

        lines = [
            f"── Win Rate Stability ({result.n_trades} trades, {len(result.windows)} windows of {window_size}) ──",
            f"  Overall WR:  {result.overall_win_rate:.1f}%",
            f"  Stability:   {grade_icon} {result.stability_grade}  (CV={result.win_rate_cv:.1f}%)",
            f"  Range:       {result.win_rate_min:.0f}% – {result.win_rate_max:.0f}%  (±{result.win_rate_std:.1f}pp)",
            f"  95% CI:      [{result.ci_lower:.1f}%, {result.ci_upper:.1f}%]",
            f"  Current:     {result.current_window_wr:.0f}%  (z={result.z_score:+.1f})",
            "",
            "  Rolling windows:",
        ]
        for w in result.windows:
            bar = "█" * int(w.win_rate / 10)
            mark = " ← current" if w.window_number == result.windows[-1].window_number else ""
            lines.append(f"    W{w.window_number:>2}  {w.win_rate:>5.0f}%  {bar}{mark}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _paydist_cmd(args: list[str]) -> str:
        """Payoff distribution: return histogram showing win/loss spread.

        Usage: /paydist [LIMIT]  (default 500 trades)
        Classifies trades into 6 buckets from Big Loss to Big Win.
        Reports skewness, kurtosis, and outlier dependence.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(15, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.payoff_distribution import compute as pd_compute

        result = pd_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 15+ trades)."

        lines = [
            f"── Payoff Distribution ({result.n_trades} trades) ──",
            f"  Shape:    {result.distribution_shape.replace('_', ' ')}",
            f"  Skewness: {result.skewness:+.2f}  Kurtosis: {result.kurtosis:+.2f}",
            f"  Mean:     {result.mean_return:+.2f}%  Median: {result.median_return:+.2f}%",
            "",
        ]
        max_n = max(b.n_trades for b in result.buckets) or 1
        for b in result.buckets:
            bar = "█" * max(1, int(b.n_trades / max_n * 15)) if b.n_trades > 0 else ""
            lines.append(
                f"  {b.label:<24} {b.n_trades:>4}tr "
                f"({b.pct_of_total:>4.0f}%)  {bar}"
            )

        lines += [
            "",
            f"  Top 10% trades contribute: {result.top10_pct_contribution:+.0f}% of total PnL",
            f"  Bottom 10% trades drag:    {result.bottom10_pct_drag:+.0f}% of total PnL",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _ruin_cmd(args: list[str]) -> str:
        """Risk of ruin: Monte Carlo probability of hitting a drawdown threshold.

        Usage: /ruin [THRESHOLD%]  (default 30%)
        Bootstraps 1000 equity paths from trade history to estimate the
        probability of losing THRESHOLD% of capital.
        """
        if conn is None:
            return "DB not wired."

        threshold = 30.0
        if args:
            try:
                threshold = max(5.0, min(float(args[0].rstrip("%")), 95.0))
            except ValueError:
                pass

        from amms.analysis.risk_of_ruin import compute as ror_compute

        result = ror_compute(conn, ruin_threshold_pct=threshold)
        if result is None:
            return "Not enough trade history (need 10+ trades)."

        risk_icon = {
            "CRITICAL": "⛔", "HIGH": "🔴", "MODERATE": "⚠️", "LOW": "✅"
        }
        ruin_pct = result.ruin_probability * 100
        level = (
            "CRITICAL" if result.ruin_probability >= 0.20 else
            "HIGH" if result.ruin_probability >= 0.10 else
            "MODERATE" if result.ruin_probability >= 0.05 else
            "LOW"
        )
        icon = risk_icon.get(level, "")

        lines = [
            f"── Risk of Ruin ({threshold:.0f}% drawdown threshold) ──",
            f"  Ruin Probability: {ruin_pct:.1f}%  {icon} [{level}]",
            f"  Based on {result.n_simulations} simulations × {result.n_trades_per_sim} trades",
            "",
            f"  Drawdown Stats (across all paths):",
            f"    Median max DD:  {result.median_max_drawdown:.1f}%",
            f"    95th pct max DD: {result.p95_max_drawdown:.1f}%",
        ]
        if result.expected_trades_to_ruin is not None:
            lines.append(f"    Median trades to ruin: {result.expected_trades_to_ruin}")

        lines += [
            "",
            f"  Historical edge:",
            f"    Win rate:  {result.win_rate:.1f}%",
            f"    Avg win:   {result.avg_win_pct:+.2f}%",
            f"    Avg loss:  {result.avg_loss_pct:+.2f}%",
            f"    Sample:    {result.n_historical_trades} trades",
            "",
            result.verdict,
        ]
        return "\n".join(lines)

    def _scorecard_cmd(args: list[str]) -> str:
        """Trading score card: weighted grade across 7 key performance metrics.

        Usage: /scorecard [LIMIT]  (default 500 trades)
        Scores win rate, profit factor, return, consistency, risk-adj return,
        expectancy, and streak risk. Returns overall grade A-F.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.scorecard import compute as sc_compute

        result = sc_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ trades)."

        grade_bar = "█" * int(result.overall_score / 10)
        grade_icon = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "🔴", "F": "⛔"}.get(result.grade, "")

        lines = [
            f"── Trading Score Card ({result.n_trades} trades) ──",
            f"  Overall: {result.grade}  {grade_icon}  {result.overall_score:.0f}/100",
            f"  [{grade_bar:<10}]",
            "",
            f"  {'Metric':<22} {'Value':>12}  {'Score':>6}  {'Grade':>5}",
            f"  {'-'*22} {'-'*12}  {'-'*6}  {'-'*5}",
        ]
        for m in result.metrics:
            bar = "█" * int(m.score / 20)
            lines.append(
                f"  {m.name:<22} {m.value:>12}  {m.score:>5.0f}%  "
                f"  {m.grade}  {bar}"
            )
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _tfreq_cmd(args: list[str]) -> str:
        """Trade frequency vs performance: does more trading help or hurt?

        Usage: /tfreq [LIMIT]  (default 500 trades)
        Buckets trading days by number of trades and shows avg return per bucket.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.trade_frequency import compute as tf_compute

        result = tf_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ trades across 5+ trading days)."

        corr_str = f"{result.correlation:.2f}" if result.correlation is not None else "n/a"
        lines = [
            f"── Trade Frequency Analysis ({result.n_trading_days} trading days, {result.n_trades} trades) ──",
            f"  Avg: {result.avg_trades_per_day:.1f} trades/day  ({result.avg_trades_per_week:.0f}/week)",
            f"  Max active day: {result.most_active_day_count} trades",
            f"  Correlation (freq vs return): {corr_str}  [{result.correlation_label}]",
            "",
            f"  {'Frequency':<18} {'Days':>5} {'Win Days%':>10} {'Avg Ret%':>9}",
            f"  {'-'*18} {'-'*5} {'-'*10} {'-'*9}",
        ]
        for b in result.buckets:
            mark = " ← best" if b.label == result.best_bucket else (" ← worst" if b.label == result.worst_bucket else "")
            lines.append(
                f"  {b.label:<18} {b.n_days:>5} "
                f"{b.win_days_pct:>9.0f}% {b.avg_daily_pnl_pct:>+9.2f}%{mark}"
            )
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _sizeperf_cmd(args: list[str]) -> str:
        """Position size vs return: do larger trades outperform?

        Usage: /sizeperf [LIMIT]  (default 500 trades)
        Buckets trades by entry value and shows avg return per tier.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.size_performance import compute as szp_compute

        result = szp_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ trades with buy_price/qty)."

        corr_str = f"{result.correlation:.2f}" if result.correlation is not None else "n/a"
        lines = [
            f"── Position Size vs Return ({result.n_trades} trades) ──",
            f"  Correlation: {corr_str}  [{result.correlation_label}]",
            "",
            f"  {'Tier':<8} {'Trades':>6} {'Avg Value':>10} {'WR':>6} {'Avg Ret%':>9}",
            f"  {'-'*8} {'-'*6} {'-'*10} {'-'*6} {'-'*9}",
        ]
        for t in result.tiers:
            mark = " ← best" if t.label == result.best_tier else (" ← worst" if t.label == result.worst_tier else "")
            lines.append(
                f"  {t.label:<8} {t.n_trades:>6} "
                f"${t.avg_position_value:>9,.0f} "
                f"{t.win_rate:>5.0f}% {t.avg_pnl_pct:>+9.2f}%{mark}"
            )
        lines += ["", result.verdict]
        return "\n".join(lines)

    def _perftrend_cmd(args: list[str]) -> str:
        """Monthly performance trend: is trading improving or declining?

        Usage: /perftrend [LIMIT]  (default 500 trades)
        Fits linear regression over monthly PnL to surface improvement/decline.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.performance_trend import compute as pt_compute

        result = pt_compute(conn, limit=limit)
        if result is None:
            return "Not enough history (need trades across 3+ calendar months)."

        dir_icon = {"improving": "↗", "declining": "↘", "flat": "→"}.get(result.trend_direction, "")
        acc_icon = {"accelerating": "⚡", "decelerating": "⚠️", "stable": "●"}.get(result.acceleration, "")

        lines = [
            f"── Monthly Performance Trend ({result.n_months} months) ──",
            f"  Trend:       {dir_icon} {result.trend_direction.upper()}  (slope {result.slope:+.0f}/month)",
            f"  R²:          {result.r_squared:.2f}",
            f"  Consistency: {result.consistency_pct:.0f}% of months profitable",
            f"  Momentum:    {acc_icon} {result.acceleration}",
            "",
            "  Monthly PnL:",
        ]
        for m in result.monthly:
            bar = "▲" if m.total_pnl >= 0 else "▼"
            mark = " ← best" if m.label == result.best_month else (" ← worst" if m.label == result.worst_month else "")
            lines.append(f"    {m.label}  {bar}  {m.total_pnl:>+10.2f}  {m.win_rate:.0f}% WR  {m.n_trades}tr{mark}")

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _symperf_cmd(args: list[str]) -> str:
        """Closed-trade performance ranked by symbol.

        Usage: /symperf [LIMIT]  (default 500 trades)
        Ranks all traded symbols by total PnL, win rate, and profit factor.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        min_trades = 2
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.symbol_performance import compute as sp_compute

        result = sp_compute(conn, limit=limit, min_trades=min_trades)
        if result is None:
            return "Not enough trade history (need 5+ trades)."

        lines = [
            f"── Symbol Performance ({result.n_symbols} symbols, {result.n_trades} trades) ──",
            f"  Total PnL: {result.total_pnl:+.2f}",
            "",
            f"  {'#':<3} {'Symbol':<8} {'Trades':>6} {'WR':>6} {'Avg':>8} {'Total':>10} {'PF':>6}",
            f"  {'-'*3} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*6}",
        ]
        for s in result.symbols:
            pf_str = f"{s.profit_factor:.2f}" if s.profit_factor is not None else "  n/a"
            mark = " ← best" if s.symbol == result.best_symbol else (" ← worst" if s.symbol == result.worst_symbol else "")
            lines.append(
                f"  {s.rank:<3} {s.symbol:<8} {s.n_trades:>6} "
                f"{s.win_rate:>5.0f}% {s.avg_pnl:>+8.2f} {s.total_pnl:>+10.2f} {pf_str:>6}{mark}"
            )

        lines += ["", result.verdict]
        return "\n".join(lines)

    def _pcal_cmd(args: list[str]) -> str:
        """Profit calendar: daily PnL summary grouped by month.

        Usage: /pcal [LIMIT]  (default 500 trades)
        Shows profit/loss per trading day as a calendar summary.
        """
        if conn is None:
            return "DB not wired."

        limit = 500
        if args:
            try:
                limit = max(10, min(int(args[0]), 2000))
            except ValueError:
                pass

        from amms.analysis.profit_calendar import compute as pc_compute

        result = pc_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ trades with sell timestamps)."

        lines = [
            f"── Profit Calendar ({result.n_months} month(s), {result.overall_profitable_days + result.overall_losing_days} trading days) ──",
            f"  Daily win rate: {result.overall_profitable_days}/{result.overall_profitable_days + result.overall_losing_days} profitable days",
            f"  Total PnL: {result.overall_pnl:+.2f}",
            "",
        ]
        for m in result.months[-3:]:  # show last 3 months
            sign = "+" if m.total_pnl >= 0 else ""
            lines.append(
                f"  {m.year}-{m.month:02d}  "
                f"{m.n_profitable_days}/{m.n_trading_days} up  "
                f"PnL:{sign}{m.total_pnl:.2f}  "
                f"avg:{m.avg_daily_pnl:+.2f}/day"
            )
            for d in m.days:
                bar = "▲" if d.total_pnl > 0 else "▼"
                lines.append(
                    f"    {d.date}  {bar}  {d.n_trades}tr  {d.total_pnl:+.2f}"
                )
            lines.append("")

        lines.append(result.verdict)
        return "\n".join(lines)

    def _holdreturn_cmd(args: list[str]) -> str:
        """Hold duration vs return analysis: which hold lengths work best?

        Usage: /holdreturn [LIMIT]  (default 200 trades)
        Splits trades into quartiles by hold duration and shows avg return %,
        win rate, and correlation between hold time and P&L.
        """
        if conn is None:
            return "DB not wired."

        limit = 200
        if args:
            try:
                limit = max(20, min(int(args[0]), 1000))
            except ValueError:
                pass

        from amms.analysis.duration_return import compute as dr_compute

        result = dr_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 10+ trades with timestamps)."

        corr_str = f"{result.correlation:.3f}" if result.correlation is not None else "n/a"
        corr_icon = "📈" if result.correlation_label == "positive" else (
            "📉" if result.correlation_label == "negative" else "↔️ "
        )

        lines = [
            f"── Hold Duration vs Return ({result.n_trades} trades) ──",
            f"  Correlation:    {corr_icon} {corr_str} ({result.correlation_label})",
            f"  Optimal range:  {result.optimal_bucket}",
            "",
            f"  {'Bucket':<18}  {'N':>4}  {'AvgDays':>7}  {'AvgRet%%':>8}  {'WR%%':>6}",
        ]
        for b in result.buckets:
            ret_icon = "🟢" if b.avg_pnl_pct >= 0 else "🔴"
            star = " ★" if b.label == result.optimal_bucket else ""
            lines.append(
                f"  {b.label:<18}  {b.n_trades:>4}  {b.avg_hold_days:>6.1f}d"
                f"  {ret_icon} {b.avg_pnl_pct:>+6.2f}%%  {b.win_rate:>5.1f}%%{star}"
            )
        lines += ["", f"  {result.verdict}"]
        return "\n".join(lines)

    def _screen_cmd(args: list[str]) -> str:
        """Screen open positions against RSI / ROC / SMA / volume criteria.

        Usage: /screen [rsi=40-70] [roc=5] [sma=above] [vol=1.5]
          rsi=MIN-MAX  — RSI range filter (e.g. rsi=40-70)
          roc=N        — min 20-bar ROC % (e.g. roc=5 for +5%%)
          sma=above    — must be above SMA20 (or below)
          vol=N        — volume ratio minimum (e.g. vol=1.5)

        Example: /screen rsi=50-70 roc=3 sma=above
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions to screen."

        # Parse args
        rsi_min = rsi_max = roc_min = vol_min = None
        above_sma = None

        for arg in args:
            if arg.startswith("rsi="):
                parts = arg[4:].split("-")
                try:
                    if len(parts) == 2:
                        rsi_min, rsi_max = float(parts[0]), float(parts[1])
                    else:
                        rsi_min = float(parts[0])
                except ValueError:
                    pass
            elif arg.startswith("roc="):
                try:
                    roc_min = float(arg[4:])
                except ValueError:
                    pass
            elif arg.startswith("sma="):
                val = arg[4:].lower()
                above_sma = True if val in ("above", "1", "true") else (False if val in ("below", "0", "false") else None)
            elif arg.startswith("vol="):
                try:
                    vol_min = float(arg[4:])
                except ValueError:
                    pass

        from amms.analysis.symbol_screener import screen as sym_screen

        # Fetch bars for all positions
        symbols_bars: dict[str, list] = {}
        for pos in positions[:20]:
            try:
                symbols_bars[pos.symbol] = data.get_bars(pos.symbol, limit=60)
            except Exception:
                pass

        result = sym_screen(
            symbols_bars,
            rsi_min=rsi_min, rsi_max=rsi_max,
            roc_min=roc_min,
            require_above_sma20=above_sma,
            volume_ratio_min=vol_min,
        )

        if result is None or not result.results:
            return "No positions matched the screening criteria."

        filter_str = "  ".join(result.filters_applied) if result.filters_applied else "none"
        lines = [
            f"── Symbol Screener ({result.n_screened} screened) ──",
            f"  Filters: {filter_str}",
            f"  Passed all: {result.n_passed}/{result.n_screened}",
            "",
            f"  {'Sym':<7}  {'Price':>8}  {'Score':>6}  {'RSI':>5}  {'ROC20':>7}  {'Vol×':>5}  {'SMA20':>6}",
        ]
        for r in result.results[:10]:
            sma_str = "↑" if r.above_sma20 is True else ("↓" if r.above_sma20 is False else " ")
            rsi_str = f"{r.rsi:.0f}" if r.rsi is not None else " n/a"
            roc_str = f"{r.roc_20:+.1f}%%" if r.roc_20 is not None else "  n/a"
            vol_str = f"{r.volume_ratio:.1f}×" if r.volume_ratio is not None else "n/a"
            lines.append(
                f"  {r.symbol:<7}  ${r.price:>7.2f}  {r.score:>5.0f}%%"
                f"  {rsi_str:>5}  {roc_str:>7}  {vol_str:>5}  {sma_str}"
            )
        return "\n".join(lines)

    def _jsum_cmd(args: list[str]) -> str:
        """Journal summary: monthly/weekly trade performance breakdown.

        Usage: /jsum [monthly|weekly] [LIMIT]
        Shows PnL, win rate, profit factor per calendar period.
        """
        if conn is None:
            return "DB not wired."

        mode = "monthly"
        limit = 500
        for arg in args:
            if arg.lower() in ("weekly", "week", "w"):
                mode = "weekly"
            elif arg.lower() in ("monthly", "month", "m"):
                mode = "monthly"
            else:
                try:
                    limit = max(10, min(int(arg), 2000))
                except ValueError:
                    pass

        from amms.analysis.journal_summary import compute as js_compute

        result = js_compute(conn, mode=mode, limit=limit)
        if result is None:
            return "Not enough trade history (need 3+ completed trades with timestamps)."

        lines = [
            f"── Trade Journal Summary ({mode}, {result.n_trades} trades) ──",
            f"  Overall PnL:  ${result.overall_pnl:+,.2f}",
            f"  Overall WR:   {result.overall_win_rate:.1f}%%",
            f"  Best period:  {result.best_period}",
            f"  Worst period: {result.worst_period}",
            "",
            f"  {'Period':<12}  {'N':>4}  {'WR%%':>6}  {'Total PnL':>10}  {'Avg':>8}  {'PF':>5}",
        ]
        for p in result.periods[-12:]:  # show last 12 periods
            pf_str = f"{p.profit_factor:.2f}" if p.profit_factor is not None else " ∞ "
            pnl_icon = "🟢" if p.total_pnl >= 0 else "🔴"
            lines.append(
                f"  {p.period:<12}  {p.n_trades:>4}  {p.win_rate:>5.1f}%%"
                f"  {pnl_icon} ${p.total_pnl:>+8.2f}  ${p.avg_pnl:>+6.2f}  {pf_str:>5}"
            )
        return "\n".join(lines)

    def _volcone_cmd(args: list[str]) -> str:
        """Historical Volatility Cone: current vol vs historical distribution.

        Usage: /volcone [SYM]
        Shows realized HV for 5/10/20/30/60-day windows with percentile rank
        and term structure (normal = calm, inverted = stress).
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /volcone AAPL)"

        from amms.analysis.volatility_cone import compute as vc_compute

        regime_icon = {"low": "🟢", "normal": "↔️ ", "elevated": "🟡", "extreme": "🔴"}
        ts_icon = {"normal": "📉", "inverted": "📈", "flat": "↔️ "}

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=300)
            except Exception:
                bars = []
            result = vc_compute(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 80+ bars)")
                continue

            ts = result.term_structure
            lines += [
                f"── {sym} Vol Cone ──",
                f"  Term structure: {ts_icon.get(ts, '')} {ts}",
                f"  Short-term:     {regime_icon.get(result.short_term_regime, '')} {result.short_term_regime}",
                f"  {'Win':>4}  {'HV':>6}  {'25th':>6}  {'Med':>6}  {'75th':>6}  {'Pct':>5}  Regime",
            ]
            for w in result.windows:
                icon = regime_icon.get(w.regime, "")
                lines.append(
                    f"  {w.window:>4}d  {w.hv:>5.1f}%%"
                    f"  {w.hv_25th:>5.1f}%%  {w.hv_median:>5.1f}%%  {w.hv_75th:>5.1f}%%"
                    f"  {w.percentile:>4.0f}%%  {icon} {w.regime}"
                )
            lines.append(f"  {result.verdict}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _ecal_cmd(args: list[str]) -> str:
        """Earnings calendar: view upcoming earnings or add dates.

        Usage:
          /ecal                        — show upcoming (7 days) for open positions
          /ecal list [DAYS]            — list all upcoming (default 30 days)
          /ecal add SYM DATE [TIME]    — add earnings date (TIME: before_open/after_close)
          /ecal remove SYM [DATE]      — remove entry
          /ecal check SYM              — check if SYM has upcoming earnings

        Example: /ecal add AAPL 2026-07-31 after_close
        """
        if conn is None:
            return "DB not wired."

        from amms.data.earnings_calendar import (
            add as ec_add, remove as ec_remove,
            upcoming as ec_upcoming, check_positions,
            ensure_table,
        )
        from datetime import date as _date

        ensure_table(conn)

        sub = args[0].lower() if args else "list"

        if sub == "add":
            if len(args) < 3:
                return "Usage: /ecal add SYM DATE [before_open|after_close]"
            sym = args[1].upper()
            dt = args[2]
            tod = args[3].lower() if len(args) >= 4 else "unknown"
            if tod not in ("before_open", "after_close", "unknown"):
                tod = "unknown"
            ok = ec_add(conn, sym, dt, time_of_day=tod)
            return f"Added {sym} earnings on {dt} ({tod})." if ok else f"Failed — invalid date: {dt}"

        if sub == "remove":
            if len(args) < 2:
                return "Usage: /ecal remove SYM [DATE]"
            sym = args[1].upper()
            dt = args[2] if len(args) >= 3 else None
            n = ec_remove(conn, sym, dt)
            return f"Removed {n} entry/entries for {sym}."

        if sub == "check":
            if len(args) < 2:
                return "Usage: /ecal check SYM"
            sym = args[1].upper()
            entries = ec_upcoming(conn, within_days=30, symbols=[sym])
            if not entries:
                return f"{sym}: no earnings in the next 30 days."
            lines = [f"{sym} upcoming earnings:"]
            for e in entries:
                lines.append(f"  {e.report_date}  ({e.time_of_day})  in {e.days_until}d")
            return "\n".join(lines)

        # Default: list or show positions earnings
        if sub == "list" or sub not in ("add", "remove", "check"):
            within = 30
            if sub == "list" and len(args) >= 2:
                try:
                    within = int(args[1])
                except ValueError:
                    pass
            elif sub not in ("list",):
                within = 7
                # Show earnings for open positions
                try:
                    positions = broker.get_positions()
                    syms = [p.symbol for p in positions]
                except Exception:
                    syms = None
                entries = check_positions(conn, syms or [], within_days=within) if syms else ec_upcoming(conn, within_days=within)
                if not entries:
                    return f"No earnings in open positions in the next {within} days."
                lines = [f"Upcoming earnings (next {within}d) for open positions:"]
                for e in entries:
                    tod_icon = {"before_open": "🌅", "after_close": "🌙", "unknown": "❓"}.get(e.time_of_day, "")
                    lines.append(f"  {e.symbol:<6}  {e.report_date}  {tod_icon} {e.time_of_day}  in {e.days_until}d")
                return "\n".join(lines)

            entries = ec_upcoming(conn, within_days=within)
            if not entries:
                return f"No earnings in the next {within} days. Add with: /ecal add AAPL 2026-07-31 after_close"
            lines = [f"Upcoming earnings (next {within}d):"]
            for e in entries:
                tod_icon = {"before_open": "🌅", "after_close": "🌙", "unknown": "❓"}.get(e.time_of_day, "")
                lines.append(f"  {e.symbol:<6}  {e.report_date}  {tod_icon} {e.time_of_day}  in {e.days_until}d")
            return "\n".join(lines)

        return "Unknown subcommand. Use /ecal list|add|remove|check"

    def _risksize_cmd(args: list[str]) -> str:
        """Fixed-fractional position sizing based on stop-loss distance.

        Usage: /risksize SYM ENTRY STOP [CAPITAL]
          SYM    — ticker symbol
          ENTRY  — planned entry price
          STOP   — stop-loss price
          CAPITAL — account size (default: uses broker account equity)

        Example: /risksize AAPL 175.00 170.00
        Shows shares for 0.5%%, 1%%, 2%% account risk with 1R/2R/3R targets.
        """
        USAGE = "Usage: /risksize SYM ENTRY STOP [CAPITAL]"

        if len(args) < 3:
            return USAGE

        sym = args[0].upper()
        try:
            entry = float(args[1])
            stop = float(args[2])
        except ValueError:
            return "Invalid price — " + USAGE

        capital = None
        if len(args) >= 4:
            try:
                capital = float(args[3])
            except ValueError:
                pass

        if capital is None:
            try:
                acc = broker.get_account()
                capital = float(acc.equity)
            except Exception:
                capital = 100_000.0

        from amms.analysis.position_sizer import compute as ps_compute

        result = ps_compute(sym, entry, stop, capital)
        if result is None:
            return (
                f"Invalid inputs: stop must be below entry, capital must be > 0.\n"
                + USAGE
            )

        lines = [
            f"── Position Sizing: {result.symbol} ──",
            f"  Entry:  ${result.entry_price:.2f}",
            f"  Stop:   ${result.stop_price:.2f}  "
            f"(${result.risk_per_share:.2f} / {result.risk_per_share_pct:.1f}%% per share)",
            f"  Capital: ${capital:,.0f}",
            "",
            f"  {'Risk%%':>5}  {'Shares':>7}  {'Value':>10}  {'Wt%%':>5}  "
            f"{'MaxLoss':>9}  1R → 2R → 3R",
        ]
        for level in result.levels:
            lines.append(
                f"  {level.risk_pct:>4.1f}%%  {level.shares:>7}  "
                f"${level.position_value:>9,.0f}  {level.position_weight_pct:>4.1f}%%"
                f"  ${level.max_loss:>8,.0f}"
                f"  ${level.target_1r:.2f} → ${level.target_2r:.2f} → ${level.target_3r:.2f}"
            )
        lines += ["", f"  {result.note}"]
        return "\n".join(lines)

    def _sectordetail_cmd(_args: list[str]) -> str:
        """Detailed sector exposure vs S&P 500 benchmark weights.

        Usage: /sectordetail
        Shows sector weights, active weight vs benchmark, HHI, and risk flags.
        Active weight > 0 = overweight vs S&P, < 0 = underweight.
        """
        from amms.analysis.sector_exposure import analyze as se_analyze

        result = se_analyze(broker)
        if result is None:
            return "No open positions."

        lines = [
            f"── Sector Exposure ({result.n_positions} positions, "
            f"{result.n_sectors} sectors) ──",
            f"  Sector HHI:  {result.portfolio_hhi:.3f}",
            f"  {result.verdict}",
            "",
            f"  {'Sector':<26}  {'Wt%%':>6}  {'Bnchmk':>7}  {'Active':>7}  {'N':>3}",
        ]
        for s in result.sectors:
            active_icon = "↑" if s.active_weight_pct > 5 else ("↓" if s.active_weight_pct < -5 else " ")
            lines.append(
                f"  {s.sector:<26}  {s.weight_pct:>5.1f}%%"
                f"  {s.benchmark_weight_pct:>6.1f}%%"
                f"  {s.active_weight_pct:>+6.1f}%% {active_icon}"
                f"  {s.n_positions:>3}"
            )
        if result.risk_flags:
            lines.append("")
            for flag in result.risk_flags:
                lines.append(f"  ⚠️  {flag}")

        return "\n".join(lines)

    def _imom_cmd(args: list[str]) -> str:
        """Intraday momentum: VWAP position, A/D ratio, session range.

        Usage: /imom [SYM]
        Shows VWAP relation, price-in-range, accumulation/distribution,
        and cumulative session return for each open position.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /imom AAPL)"

        from amms.analysis.intraday_momentum import compute as imom_compute

        signal_icon = {
            "strong_bull": "🚀", "bull": "🟢",
            "neutral": "↔️ ", "bear": "🔴", "strong_bear": "⛔",
        }

        lines = ["Intraday Momentum (VWAP / A-D / session range):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=40)
            except Exception:
                bars = []
            result = imom_compute(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 3+ bars)")
                continue
            icon = signal_icon.get(result.momentum_signal, "")
            fade = " fade⚠️" if result.fade_detected else ""
            lines.append(
                f"  {sym:<6}  ${result.current_price:.2f}"
                f"  VWAP ${result.vwap:.2f} [{result.price_vs_vwap}]"
                f"  rng {result.price_in_range_pct:.0f}%%"
                f"  ret {result.cumulative_return_pct:+.1f}%%"
                f"  {icon} {result.momentum_signal}{fade}"
            )
        return "\n".join(lines)

    def _dduration_cmd(args: list[str]) -> str:
        """Drawdown duration analysis: how long and how deep.

        Usage: /dduration [PERIODS]  (default 252 snapshots)
        Shows current drawdown depth and duration, max historical drawdown,
        average recovery time, and pain index from equity curve.
        """
        if conn is None:
            return "DB not wired."

        limit = 252
        if args:
            try:
                limit = max(20, min(int(args[0]), 1000))
            except ValueError:
                pass

        from amms.analysis.drawdown_duration import compute as dd_compute

        result = dd_compute(conn, limit=limit)
        if result is None:
            return "Not enough equity history (need 10+ snapshots)."

        status_icon = "🔴" if result.is_underwater else "🟢"

        lines = [
            f"── Drawdown Duration ({result.n_periods} snapshots) ──",
            f"  Status:        {status_icon} {'Underwater' if result.is_underwater else 'At new high'}",
        ]

        if result.is_underwater:
            lines += [
                f"  Current DD:    {result.current_drawdown_pct:.2f}%%",
                f"  DD duration:   {result.current_drawdown_periods} periods",
            ]

        lines += [
            f"  Equity high:   ${result.equity_high:,.0f}",
            f"  Current equity: ${result.current_equity:,.0f}",
            "",
            f"  Max drawdown:  {result.max_drawdown_pct:.2f}%%  ({result.max_drawdown_duration} periods)",
            f"  Longest uw:    {result.longest_underwater} periods",
            f"  Pain index:    {result.pain_index:.3f}%%  (avg drawdown)",
            f"  Episodes:      {result.n_drawdown_periods} total, {result.n_recovered} recovered",
        ]

        if result.avg_recovery_periods is not None:
            lines.append(f"  Avg recovery:  {result.avg_recovery_periods:.1f} periods")

        lines += ["", f"  {result.verdict}"]
        return "\n".join(lines)

    def _expectancy_cmd(args: list[str]) -> str:
        """Trade expectancy per trade and per symbol.

        Usage: /expectancy [LIMIT]  (default 200 recent trades)
        Shows expected $ per trade, R-multiple expectancy, and per-symbol breakdown.
        Expectancy > 0 = positive edge. Grade A = 0.5R+.
        """
        if conn is None:
            return "DB not wired."

        limit = 200
        if args:
            try:
                limit = max(10, min(int(args[0]), 1000))
            except ValueError:
                pass

        from amms.analysis.expectancy import compute as exp_compute

        result = exp_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ completed trades)."

        grade_icon = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}.get(
            result.overall.grade, ""
        )

        o = result.overall
        lines = [
            f"── Trade Expectancy ({result.n_trades} trades) ──",
            f"  Win rate:   {o.win_rate:.1f}%%",
            f"  Avg win:    ${o.avg_win:.2f}",
            f"  Avg loss:   ${o.avg_loss:.2f}",
            f"  Expectancy: ${o.expectancy:+.2f} per trade",
            f"  R-multiple: {o.r_expectancy:+.3f}R per trade",
            f"  Grade:      {grade_icon} {o.grade}",
        ]

        if result.by_symbol:
            lines += [
                "",
                f"  Per-symbol breakdown ({result.n_symbols} symbols):",
                f"  {'Symbol':<8}  {'N':>4}  {'WR%%':>6}  {'Exp$':>8}  {'R':>6}  Grade",
            ]
            for se in result.by_symbol[:8]:
                g_icon = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}.get(se.grade, "")
                lines.append(
                    f"  {se.symbol:<8}  {se.n_trades:>4}  {se.win_rate:>5.1f}%%"
                    f"  ${se.expectancy:>+7.2f}  {se.r_expectancy:>+5.3f}R  {g_icon} {se.grade}"
                )
            if result.best_symbol:
                lines.append(f"  Best: {result.best_symbol}  |  Worst: {result.worst_symbol}")

        return "\n".join(lines)

    def _vpdetail_cmd(args: list[str]) -> str:
        """Detailed volume profile: POC, Value Area, HVN/LVN nodes.

        Usage: /vpdetail [SYM]
        Shows Point of Control, 70% Value Area, high/low volume nodes,
        and nearest support/resistance levels from volume distribution.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /vpdetail AAPL)"

        from amms.analysis.volume_profile import compute as vp_compute

        lines = []
        for sym in symbols[:4]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = vp_compute(bars, symbol=sym)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 10+ bars with volume)")
                continue

            va_icon = {
                "above_va": "⬆️ ",
                "in_va":    "↔️ ",
                "below_va": "⬇️ ",
            }.get(result.price_vs_va, "")

            lines += [
                f"── {sym} Volume Profile ({result.bars_used} bars) ──",
                f"  Price:    ${result.current_price:.2f}  {va_icon} {result.price_vs_va}",
                f"  POC:      ${result.poc:.2f}  ({result.price_vs_poc})",
                f"  VAH:      ${result.vah:.2f}",
                f"  VAL:      ${result.val:.2f}",
                f"  HVNs: {result.hvn_count}  LVNs: {result.lvn_count}",
            ]
            if result.nearest_hvn_above:
                lines.append(f"  HVN above: ${result.nearest_hvn_above:.2f}")
            if result.nearest_hvn_below:
                lines.append(f"  HVN below: ${result.nearest_hvn_below:.2f}")
            if result.nearest_lvn_above:
                lines.append(f"  LVN above: ${result.nearest_lvn_above:.2f}  (gap zone)")
            if result.nearest_lvn_below:
                lines.append(f"  LVN below: ${result.nearest_lvn_below:.2f}  (gap zone)")
            lines.append(f"  {result.verdict}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _tstreak_cmd(args: list[str]) -> str:
        """Trade streak analysis: win/loss streaks from trade history.

        Usage: /tstreak [LIMIT]  (default 100 recent trades)
        Shows current streak, longest streaks, recent form, and tilt risk.
        """
        if conn is None:
            return "DB not wired."

        limit = 100
        if args:
            try:
                limit = max(10, min(int(args[0]), 500))
            except ValueError:
                pass

        from amms.analysis.trade_streak import compute as streak_compute

        result = streak_compute(conn, limit=limit)
        if result is None:
            return "Not enough trade history (need 5+ completed trades)."

        streak_icon = "🔥" if result.hot_hand else ("🧊" if result.tilt_risk else "↔️ ")
        form_icon = {
            "hot": "🔥", "warm": "🟢", "cool": "🟡", "icy": "🔵",
        }.get(result.recent_form_label, "")
        mom_icon = {
            "improving": "📈", "declining": "📉", "stable": "↔️ ",
        }.get(result.momentum, "")

        lines = [
            f"── Trade Streak ({result.n_trades} trades) ──",
            f"  Current streak:   {streak_icon} {result.current_streak_label}",
            f"  Longest win run:  {result.longest_win_streak}W",
            f"  Longest loss run: {result.longest_loss_streak}L",
            "",
            f"  Recent form (last 10): {result.recent_form:.0f}%%  {form_icon} {result.recent_form_label}",
            f"  Momentum:              {mom_icon} {result.momentum}",
            f"  Overall win rate:      {result.overall_win_rate:.1f}%%",
            "",
            f"  {result.verdict}",
        ]
        return "\n".join(lines)

    def _riskratios_cmd(args: list[str]) -> str:
        """Risk-adjusted performance: Sharpe, Sortino, Calmar.

        Usage: /riskratios [PERIODS]  (default 252 snapshots)
        Computes Sharpe, Sortino, Calmar ratios from equity curve history.
        Grade: excellent (≥2) / good (≥1) / ok (≥0) / poor (<0).
        """
        if conn is None:
            return "DB not wired."

        limit = 252
        if args:
            try:
                limit = max(20, min(int(args[0]), 1000))
            except ValueError:
                pass

        from amms.analysis.risk_ratios import compute as rr_compute

        result = rr_compute(conn, limit=limit)
        if result is None:
            return "Not enough equity history (need 10+ snapshots in equity_snapshots)."

        def _fmt_ratio(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "n/a"

        grade_icon = {
            "excellent": "🟢", "good": "🟡", "ok": "🟠", "poor": "🔴",
        }.get(result.sharpe_grade, "")

        lines = [
            f"── Risk-Adjusted Performance ({result.n_periods} snapshots) ──",
            f"  Equity:        ${result.start_equity:,.0f} → ${result.end_equity:,.0f}",
            f"  Ann. return:   {result.annualized_return_pct:+.2f}%%",
            f"  Ann. vol:      {result.annualized_vol_pct:.2f}%%",
            f"  Downside vol:  {result.annualized_downside_vol_pct:.2f}%%",
            f"  Max drawdown:  {result.max_drawdown_pct:.2f}%%",
            "",
            f"  Sharpe:  {_fmt_ratio(result.sharpe):>7}  {grade_icon} {result.sharpe_grade}",
            f"  Sortino: {_fmt_ratio(result.sortino):>7}",
            f"  Calmar:  {_fmt_ratio(result.calmar):>7}",
            "",
            f"  {result.verdict}",
        ]
        return "\n".join(lines)

    def _watch_cmd(args: list[str]) -> str:
        """Watchlist manager: add, remove, list, scan symbols.

        Usage:
          /watch add AAPL [note]  — add symbol
          /watch remove AAPL     — remove symbol
          /watch list            — show all symbols
          /watch scan            — price scan of watchlist (requires data)
        """
        if conn is None:
            return "DB not wired."

        from amms.data.watchlist import add as wl_add, remove as wl_remove
        from amms.data.watchlist import list_all, contains

        if not args:
            entries = list_all(conn)
            if not entries:
                return "Watchlist is empty. Use /watch add AAPL"
            lines = [f"── Watchlist ({len(entries)} symbols) ──"]
            for e in entries:
                note_str = f"  — {e.note}" if e.note else ""
                lines.append(f"  {e.symbol:<8} added {e.added_ts[:10]}{note_str}")
            return "\n".join(lines)

        sub = args[0].lower()

        if sub == "add":
            if len(args) < 2:
                return "usage: /watch add SYMBOL [note]"
            sym = args[1].upper()
            note = " ".join(args[2:]) if len(args) > 2 else ""
            if wl_add(conn, sym, note=note):
                return f"{sym} added to watchlist."
            return f"{sym} is already on the watchlist."

        elif sub in ("remove", "rm", "del"):
            if len(args) < 2:
                return "usage: /watch remove SYMBOL"
            sym = args[1].upper()
            if wl_remove(conn, sym):
                return f"{sym} removed from watchlist."
            return f"{sym} was not on the watchlist."

        elif sub == "list":
            entries = list_all(conn)
            if not entries:
                return "Watchlist is empty."
            return "\n".join(f"  {e.symbol}" for e in entries)

        elif sub == "scan":
            if data is None:
                return "Data client not wired."
            entries = list_all(conn)
            if not entries:
                return "Watchlist is empty."
            lines = [f"── Watchlist Scan ({len(entries)} symbols) ──"]
            for e in entries:
                try:
                    bars = data.get_bars(e.symbol, limit=3)
                    if bars and len(bars) >= 2:
                        curr = bars[-1].close
                        prev = bars[-2].close
                        chg = (curr - prev) / prev * 100 if prev > 0 else 0.0
                        lines.append(
                            f"  {e.symbol:<8} ${curr:.2f}  {chg:+.2f}%%"
                        )
                    else:
                        lines.append(f"  {e.symbol:<8} (no data)")
                except Exception as ex:
                    lines.append(f"  {e.symbol:<8} error: {ex!r}")
            return "\n".join(lines)

        else:
            return (
                "Unknown subcommand. Use: add | remove | list | scan\n"
                "Example: /watch add AAPL"
            )

    def _trendq_cmd(args: list[str]) -> str:
        """Trend quality/consistency score for open positions.

        Usage: /trendq [SYM] [LOOKBACK]
        Measures R², trend efficiency, and noise level.
        Default lookback: 20 bars.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        lookback = 20
        sym_filter = None
        for arg in args:
            if arg.isdigit():
                lookback = max(5, min(int(arg), 60))
            else:
                sym_filter = arg.upper()

        symbols = ([sym_filter] if sym_filter
                   else [p.symbol for p in positions])
        if not symbols:
            return "no open positions (pass a ticker: /trendq AAPL)"

        from amms.analysis.trend_consistency import score as tc_score

        LABEL_ICON = {
            "consistent": "▲▲",
            "moderate": "▲",
            "choppy": "↕",
            "random": "?",
        }

        lines = [f"── Trend Quality ({lookback}d) ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=lookback + 5)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = tc_score(bars, lookback=lookback)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            icon = LABEL_ICON.get(result.label, "?")
            dir_arrow = "↑" if result.direction == "up" else ("↓" if result.direction == "down" else "→")
            lines.append(
                f"  {icon} {sym:<6}  {result.score:.0f}/100 [{result.label}]"
                f"  {dir_arrow} R² {result.r_squared:.2f}"
                f"  eff {result.efficiency:.2f}"
                f"  noise {result.noise_pct:.1f}%%"
            )

        return "\n".join(lines)

    def _overnight_cmd(args: list[str]) -> str:
        """Overnight gap risk for open positions.

        Usage: /overnight [SYM]
        Analyzes historical open-vs-prev-close gaps to quantify
        overnight holding risk.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /overnight AAPL)"

        from amms.analysis.overnight_risk import analyze as or_analyze

        LABEL_ICON = {
            "low": "✓",
            "moderate": "→",
            "elevated": "⚠",
            "high": "✗",
        }

        lines = ["── Overnight Gap Risk ──"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=40)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = or_analyze(bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data")
                continue

            icon = LABEL_ICON.get(result.risk_label, "?")
            lines.append(
                f"  [{icon}] {sym:<6}  {result.risk_label:<8}"
                f"  score {result.risk_score:.0f}"
                f"  freq {result.gap_frequency_pct:.0f}%%"
                f"  avg {result.avg_gap_pct:.1f}%%"
                f"  max {result.max_gap_pct:.1f}%%"
            )

        return "\n".join(lines)

    def _rsrank_cmd(args: list[str]) -> str:
        """Relative strength ranking for open positions.

        Usage: /rsrank [LOOKBACK]
        Ranks positions by return vs portfolio-average benchmark.
        Default lookback: 20 bars.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions."

        lookback = 20
        if args and args[0].isdigit():
            lookback = max(5, min(int(args[0]), 90))

        bars_map: dict[str, list] = {}
        for pos in positions:
            try:
                bars = data.get_bars(pos.symbol, limit=lookback + 5)
                if bars:
                    bars_map[pos.symbol] = bars
            except Exception:
                pass

        from amms.analysis.relative_strength import rank as rs_rank

        result = rs_rank(bars_map, lookback=lookback)
        if result is None:
            return "Insufficient data to rank positions."

        TREND_ICON = {
            "outperforming": "▲",
            "neutral": "→",
            "underperforming": "▼",
        }

        lines = [
            f"── Relative Strength Ranking ({lookback}d) ──",
            f"  Benchmark (portfolio avg): {result.benchmark_return_pct:+.1f}%%",
            "",
            f"  {'#':<3}  {'Sym':<6}  {'RS':>5}  {'Abs%%':>7}  {'Rel%%':>7}",
        ]
        for rank_i, row in enumerate(result.rows, 1):
            icon = TREND_ICON.get(row.trend, "?")
            lines.append(
                f"  {rank_i:<3}  {row.symbol:<6}  {row.rs_score:>5.0f}"
                f"  {row.abs_return_pct:>+6.1f}%%"
                f"  {row.rel_return_pct:>+6.1f}%%  {icon}"
            )

        return "\n".join(lines)

    def _stopopt_cmd(args: list[str]) -> str:
        """ATR-based stop loss optimizer for open positions.

        Usage: /stopopt [SYM]
        Suggests conservative/standard/wide stops and a 2R target.
        Based on ATR-14.  Warns if current price has violated the stop.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        if not positions:
            return "No open positions."

        # Filter to requested symbol or all
        if args:
            sym_filter = args[0].upper()
            positions = [p for p in positions if p.symbol == sym_filter]
            if not positions:
                return f"{sym_filter} not in open positions."

        from amms.analysis.stop_optimizer import suggest_stops

        lines = ["── Stop Loss Optimizer (ATR-14) ──"]
        for pos in positions[:8]:
            sym = pos.symbol
            try:
                entry = float(pos.avg_entry_price)
            except Exception:
                entry = 0.0
            try:
                bars = data.get_bars(sym, limit=25)
            except Exception as e:
                lines.append(f"  {sym}: data error {e!r}")
                continue

            result = suggest_stops(sym, entry, bars)
            if result is None:
                lines.append(f"  {sym}: insufficient data (need 15+ bars)")
                continue

            warn = "  ⚠️ STOP VIOLATED" if result.stop_violated else ""
            lines.append(
                f"  {sym}  entry ${result.entry_price:.2f}"
                f"  curr ${result.current_price:.2f}"
                f"  ATR {result.atr_pct:.1f}%%{warn}"
            )
            lines.append(
                f"    Conservative: ${result.stop_conservative:.2f}"
                f"  Standard: ${result.stop_standard:.2f}"
                f"  Wide: ${result.stop_wide:.2f}"
            )
            lines.append(
                f"    Risk (std): {result.risk_standard_pct:.1f}%%"
                f"  2R target: ${result.target_2r:.2f}"
            )

        return "\n".join(lines)

    def _holdtime_cmd(args: list[str]) -> str:
        """Win rate and P&L by holding period.

        Usage: /holdtime [N]
        Groups closed trades into day/swing/medium/long buckets and
        shows win rate, avg P&L, and profit factor per bucket.
        Requires a connected DB with trade_pairs table.
        """
        if conn is None:
            return "DB not wired."

        limit = 200
        if args and args[0].isdigit():
            limit = max(10, min(int(args[0]), 500))

        from amms.analysis.hold_time_analysis import compute as ht_compute

        report = ht_compute(conn, limit=limit)
        if report is None:
            return "No completed trades with timestamps found."

        lines = [
            f"── Hold Time Analysis ({report.total_trades} trades) ──",
            f"  Overall win rate: {report.overall_win_rate:.1f}%%",
            f"  Best bucket:      {report.best_bucket or 'n/a'}",
            "",
            f"  {'Bucket':<8}  {'N':>4}  {'WR%%':>6}  {'Avg PnL':>9}  "
            f"{'Avg %%':>7}  {'PF':>5}",
        ]
        for b in report.buckets:
            pf_str = f"{b.profit_factor:.2f}" if b.profit_factor is not None else "∞"
            star = " *" if b.bucket == report.best_bucket else ""
            lines.append(
                f"  {b.label:<8}  {b.n_trades:>4}  {b.win_rate:>5.1f}%%"
                f"  ${b.avg_pnl:>8.2f}  {b.avg_pnl_pct:>6.1f}%%  {pf_str:>5}{star}"
            )

        return "\n".join(lines)

    def _swings_cmd(args: list[str]) -> str:
        """Swing high/low detection: key pivot levels, trend, stop, target.

        Usage: /swings [SYM]
        Detects pivot highs/lows, classifies trend, suggests stop/target.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /swings AAPL)"

        from amms.features.swing_points import detect_swings

        lines = ["Swing Highs/Lows (pivot detection):"]
        for sym in symbols[:6]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = detect_swings(bars, window=3)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 9+ bars)")
                continue
            trend_icon = {
                "uptrend": "📈", "downtrend": "📉",
                "sideways": "↔️ ", "unknown": "❓",
            }.get(result.trend, "")
            lines.append(
                f"  {sym:<6}  ${result.current_price:.2f}  "
                f"{trend_icon} {result.trend}"
            )
            if result.last_swing_high:
                brk = "  🚀 BREAKOUT" if result.breakout_up else ""
                lines.append(f"    Last swing high: ${result.last_swing_high.price:.2f}{brk}")
            if result.last_swing_low:
                bdn = "  📉 BREAKDOWN" if result.breakdown_down else ""
                lines.append(f"    Last swing low:  ${result.last_swing_low.price:.2f}{bdn}")
            if result.stop_below:
                lines.append(f"    Suggested stop:  ${result.stop_below:.2f}")
            if result.target_above:
                lines.append(f"    Target:          ${result.target_above:.2f}")
        return "\n".join(lines)

    def _aging_cmd(args: list[str]) -> str:
        """Position aging report: hold time, style, P&L, overstay flags.

        Usage: /aging
        Shows how long each position has been held and flags potential overstays.
        """
        from amms.analysis.position_aging import analyze_aging

        result = analyze_aging(broker, conn=conn)
        if result is None:
            return "No open positions."

        style_icon = {
            "day": "⚡", "swing": "📊", "medium": "📈", "long": "🏔️", "unknown": "❓",
        }

        lines = [
            f"── Position Aging ({result.total_positions} positions) ──",
        ]
        if result.avg_hold_days is not None:
            lines.append(f"  Avg hold: {result.avg_hold_days:.0f} days")
        if result.overstayed_count > 0:
            lines.append(f"  ⚠️  Overstayed: {result.overstayed_count}")
        lines.append("")

        for pos in result.positions:
            days_str = f"{pos.hold_days}d" if pos.hold_days is not None else "n/a"
            pnl_str = f"{pos.pnl_pct:+.1f}%%" if pos.pnl_pct else "n/a"
            icon = style_icon.get(pos.hold_style, "")
            flag = "  ⚠️ " + pos.overstay_reason if pos.overstay_flag else ""
            lines.append(f"  {pos.symbol:<6}  {days_str:>5}  {pnl_str:>8}  {icon} {pos.hold_style}{flag}")

        return "\n".join(lines)

    def _corrmatrix_cmd(args: list[str]) -> str:
        """Portfolio correlation matrix: pairwise return correlation + diversity score.

        Usage: /corrmatrix
        Identifies highly correlated pairs (concentration risk) and diversification quality.
        """
        if data is None:
            return "Data client not wired."

        from amms.analysis.correlation_matrix import compute as compute_corr

        result = compute_corr(broker, data)
        if result is None:
            return "Need at least 2 positions with sufficient bar history."

        level_icon = {
            "very_high": "🔴", "high": "🟠", "moderate": "🟡",
            "low": "🟢", "negative": "💚",
        }

        lines = [
            f"── Correlation Matrix ({len(result.symbols)} positions, {result.n_bars} bars) ──",
            f"  Avg correlation:     {result.avg_correlation:.2f}",
            f"  Diversification:     {result.diversification_score:.0f}/100",
        ]

        if result.high_corr_pairs:
            lines.append(f"  ⚠️  High-corr pairs ({len(result.high_corr_pairs)}):")
            for p in result.high_corr_pairs:
                lines.append(f"    {p.sym1} ↔ {p.sym2}: {p.correlation:+.2f}")

        lines.append("")
        lines.append("  All pairs:")
        for p in result.pairs:
            icon = level_icon.get(p.level, "")
            lines.append(f"  {icon} {p.sym1:<6} ↔ {p.sym2:<6}  {p.correlation:+.3f}  [{p.level}]")

        return "\n".join(lines)

    def _gaps_cmd(args: list[str]) -> str:
        """Gap analysis: recent price gaps, fill status, and gap S/R levels.

        Usage: /gaps [SYM]
        Shows significant opening gaps, whether they were filled, and gap zones.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /gaps AAPL)"

        from amms.features.gap import analyze_gaps

        lines = ["Gap Analysis (significant opening gaps):"]
        for sym in symbols[:6]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = analyze_gaps(bars, min_gap_pct=0.3)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 3+ bars)")
                continue
            lines.append(f"  {sym:<6}  ${result.current_price:.2f}  "
                         f"{len(result.gaps)} gaps  {len(result.unfilled_gaps)} unfilled")
            if result.nearest_gap_support is not None:
                lines.append(f"    Gap support:    ${result.nearest_gap_support:.2f}")
            if result.nearest_gap_resistance is not None:
                lines.append(f"    Gap resistance: ${result.nearest_gap_resistance:.2f}")
            if result.last_gap:
                g = result.last_gap
                fill_str = "✓ filled" if g.filled else "○ open"
                lines.append(
                    f"    Last gap: {g.direction} {g.gap_pct:.1f}%%  "
                    f"(prev close ${g.prev_close:.2f} → open ${g.open_price:.2f})  {fill_str}"
                )
        return "\n".join(lines)

    def _sectorheat_cmd(args: list[str]) -> str:
        """Sector momentum heatmap: 5d/20d/60d returns for all 11 sectors.

        Usage: /sectorheat
        Ranks sectors by composite momentum score. Hot = outperforming.
        """
        if data is None:
            return "Data client not wired."

        from amms.analysis.sector_rotation import sector_heatmap

        rows = sector_heatmap(data)
        if not rows:
            return "No sector data available."

        heat_icon = {
            "hot": "🔥", "warm": "🟢", "flat": "↔️ ", "cool": "🟡", "cold": "🔵", "n/a": "❓",
        }

        lines = ["── Sector Heatmap (5d / 20d / 60d momentum) ──"]
        for row in rows:
            m5 = f"{row.mom_5d:+.1f}%%" if row.mom_5d is not None else "  n/a"
            m20 = f"{row.mom_20d:+.1f}%%" if row.mom_20d is not None else "  n/a"
            m60 = f"{row.mom_60d:+.1f}%%" if row.mom_60d is not None else "  n/a"
            icon = heat_icon.get(row.trend_20d, "")
            lines.append(
                f"  {icon} {row.sector:<22}  "
                f"5d {m5:>7}  20d {m20:>7}  60d {m60:>7}  "
                f"score {row.composite_score:+.1f}"
            )
        return "\n".join(lines)

    def _btstats_cmd(args: list[str]) -> str:
        """Extended backtest statistics from the last backtest run.

        Usage: /btstats [DAYS]
        Shows Calmar, Sortino, recovery factor, streaks, expectancy, payoff ratio.
        Requires /backtest to be wired.
        """
        if run_backtest is None:
            return "backtest not wired (only available when launched from scheduler)"
        try:
            days = int(args[0]) if args else 90
        except ValueError:
            return "usage: /btstats [DAYS]  (default 90)"
        days = max(5, min(days, 730))
        try:
            bt_result = run_backtest(days)
        except Exception as e:
            return f"backtest failed: {e!r}"
        if bt_result is None:
            return "backtest produced no result"

        try:
            from amms.backtest.stats import compute_extended_stats
            ext = compute_extended_stats(bt_result._result, bt_result) if hasattr(bt_result, "_result") else None
        except Exception:
            ext = None

        if ext is None:
            return "Extended stats not available (backtest result format mismatch)."

        lines = [
            f"── Extended Backtest Stats ({days}d) ──",
            f"  Calmar ratio:    {ext.calmar_ratio:.2f}  (ann. return / max DD)",
            f"  Sortino ratio:   {ext.sortino_ratio:.2f}  (ann. return / downside σ)",
            f"  Recovery factor: {ext.recovery_factor:.2f}  (total return / max DD)",
            f"  Payoff ratio:    {ext.payoff_ratio:.2f}  (avg win / avg loss)",
            f"  Expectancy:      ${ext.expectancy:.2f}  per trade",
            f"  Max win streak:  {ext.max_consec_wins}",
            f"  Max loss streak: {ext.max_consec_losses}",
            f"  Tail ratio:      {ext.tail_ratio:.2f}  (95th / 5th pctile return)",
        ]
        return "\n".join(lines)

    def _meanrev_cmd(args: list[str]) -> str:
        """Mean reversion score: how stretched is the price from its mean?

        Usage: /meanrev [SYM]
        Aggregates Bollinger %B, Z-score, RSI deviation, Williams %%R.
        Score 0-100: extreme stretch → high probability of mean reversion.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /meanrev AAPL)"

        from amms.analysis.mean_reversion import score as mr_score

        lines = ["Mean Reversion Score (0=near mean, 100=extreme stretch):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                bars = []
            result = mr_score(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 20+ bars)")
                continue
            dir_icon = {
                "bullish_reversion": "🟢 BUY",
                "bearish_reversion": "🔴 SELL",
                "neutral": "↔️  HOLD",
            }.get(result.direction, "")
            verdict_icon = {
                "extreme": "🚨", "strong": "⚠️ ", "moderate": "📊",
                "mild": "📉", "none": "➖",
            }.get(result.verdict, "")
            lines.append(
                f"  {sym:<6}  score {result.score:.0f}/100  "
                f"{verdict_icon} {result.verdict}  {dir_icon}"
            )
            lines.append(f"         {result.recommended_action}")
        return "\n".join(lines)

    def _breadth_cmd(args: list[str]) -> str:
        """Portfolio breadth: how many positions are technically healthy?

        Usage: /breadth
        Shows VWAP, RSI, SMA-20, OBV breadth per position + overall score.
        """
        if data is None:
            return "Data client not wired."

        from amms.analysis.market_breadth import analyze_breadth

        result = analyze_breadth(broker, data)
        if result is None:
            return "No positions or insufficient data for breadth analysis."

        verdict_icon = {
            "strong": "🟢", "moderate": "🟡",
            "weak": "🟠", "deteriorating": "🔴",
        }.get(result.verdict, "")

        lines = [
            f"── Portfolio Breadth ({result.n_positions} positions) ──",
            f"  Above VWAP:  {result.pct_above_vwap:.0f}%%",
            f"  RSI > 50:    {result.pct_rsi_above_50:.0f}%%",
            f"  Above SMA20: {result.pct_above_sma20:.0f}%%",
            f"  OBV rising:  {result.pct_obv_rising:.0f}%%",
            f"  Overall:     {result.overall_score:.0f}%%  {verdict_icon} {result.verdict}",
            "",
        ] + result.detail
        return "\n".join(lines)

    def _trendlines_cmd(args: list[str]) -> str:
        """Auto-detect support and resistance trend lines.

        Usage: /trendlines [SYM]
        Fits lines through pivot highs/lows to show trend direction and breakout risk.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /trendlines AAPL)"

        from amms.analysis.trendlines import detect_trendlines

        lines = ["Trend Lines (auto-detected support/resistance):"]
        for sym in symbols[:6]:
            try:
                bars = data.get_bars(sym, limit=80)
            except Exception:
                bars = []
            result = detect_trendlines(bars)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 10+ bars)")
                continue
            pat_icon = {
                "uptrend": "📈", "downtrend": "📉",
                "wedge": "⚠️ ", "ranging": "↔️ ", "unknown": "❓",
            }.get(result.pattern, "")
            lines.append(f"  {sym:<6}  price ${result.current_price:.2f}  {pat_icon} {result.pattern}")
            if result.support:
                dist = f"  (+{result.support_distance_pct:.1f}%)" if result.support_distance_pct is not None else ""
                lines.append(
                    f"    Support:    ${result.support.current_value:.2f}  "
                    f"{result.support.direction}  slope {result.support.slope:+.3f}{dist}"
                )
            if result.resistance:
                dist = f"  ({result.resistance_distance_pct:.1f}% gap)" if result.resistance_distance_pct is not None else ""
                lines.append(
                    f"    Resistance: ${result.resistance.current_value:.2f}  "
                    f"{result.resistance.direction}  slope {result.resistance.slope:+.3f}{dist}"
                )
        return "\n".join(lines)

    def _roc_cmd(args: list[str]) -> str:
        """Rate of Change at 10/20/50 bars for a ticker or open positions.

        Usage: /roc [SYM]
        Shows short/medium/long-term momentum and overall trend direction.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /roc AAPL)"

        from amms.features.roc import multi_roc

        lines = ["Rate of Change (10/20/50-bar momentum):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=60)
            except Exception:
                bars = []
            result = multi_roc(bars)
            parts = []
            if result.short:
                parts.append(f"10d {result.short.value:+.1f}%%")
            if result.medium:
                parts.append(f"20d {result.medium.value:+.1f}%%")
            if result.long:
                parts.append(f"50d {result.long.value:+.1f}%%")
            if not parts:
                lines.append(f"  {sym:<6}  n/a (need 11+ bars)")
                continue
            trend_icon = {
                "accelerating_up": "🚀",
                "accelerating_down": "📉",
                "mixed": "↕️ ",
                "decelerating": "⚠️ ",
                "flat": "↔️ ",
            }.get(result.overall, "")
            lines.append(f"  {sym:<6}  {' | '.join(parts)}  {trend_icon} {result.overall}")
        return "\n".join(lines)

    def _wr_cmd(args: list[str]) -> str:
        """Williams %%R oscillator for a ticker or open positions.

        Usage: /wr [SYM]
        Range 0 to -100. Above -20 = overbought; below -80 = oversold.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /wr AAPL)"

        from amms.features.williams_r import williams_r

        lines = ["Williams %%R (0 to -100 | overbought>-20, oversold<-80):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                bars = []
            result = williams_r(bars, period=14, smooth=3)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 17+ bars)")
                continue
            zone_icon = {"overbought": "🔴", "oversold": "🟢", "neutral": "↔️ "}.get(result.zone, "")
            lines.append(
                f"  {sym:<6}  %%R {result.value:.1f}  smooth {result.smoothed:.1f}  "
                f"{zone_icon} {result.zone}  → {result.signal}"
            )
        return "\n".join(lines)

    def _cci_cmd(args: list[str]) -> str:
        """Commodity Channel Index for a ticker or open positions.

        Usage: /cci [SYM]
        Above +100 = overbought; below -100 = oversold.
        """
        if data is None:
            return "Data client not wired."
        try:
            positions = broker.get_positions()
        except Exception as e:
            return f"broker error: {e!r}"

        symbols = [args[0].upper()] if args else [p.symbol for p in positions]
        if not symbols:
            return "no open positions (pass a ticker: /cci AAPL)"

        from amms.features.cci import cci as compute_cci

        lines = ["CCI (Commodity Channel Index | overbought>+100, oversold<-100):"]
        for sym in symbols[:8]:
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                bars = []
            result = compute_cci(bars, period=20)
            if result is None:
                lines.append(f"  {sym:<6}  n/a (need 20+ bars)")
                continue
            zone_icon = {"overbought": "🔴", "oversold": "🟢", "neutral": "↔️ "}.get(result.zone, "")
            lines.append(
                f"  {sym:<6}  CCI {result.value:+.1f}  "
                f"{zone_icon} {result.zone}  → {result.signal}"
            )
        return "\n".join(lines)

    def _optimize(args: list[str]) -> str:
        """Show recommended portfolio allocation weights.

        Usage:
          /optimize                   — equal-weight allocation
          /optimize momentum          — momentum-score-weighted allocation
          /optimize inversevol        — inverse-volatility (risk-parity) allocation
        """
        if data is None:
            return "Data client not wired."

        mode_arg = args[0].lower().replace("-", "_") if args else "equal_weight"
        mode_map = {
            "equal": "equal_weight",
            "equal_weight": "equal_weight",
            "momentum": "momentum",
            "mom": "momentum",
            "inversevol": "inverse_vol",
            "inverse_vol": "inverse_vol",
            "riskparity": "inverse_vol",
        }
        mode = mode_map.get(mode_arg)
        if mode is None:
            return f"Unknown mode '{mode_arg}'. Use: equal, momentum, inversevol"

        symbols: list[str] = list(static_watchlist)
        if get_wsb_extras is not None:
            symbols += list(get_wsb_extras())
        if db_path is not None:
            from amms.data.dynamic_watchlist import load as load_dyn
            symbols += list(load_dyn(db_path))
        symbols = list(dict.fromkeys(s.upper() for s in symbols))

        if not symbols:
            return "Watchlist is empty. Add tickers with /add SYM."

        try:
            equity = float(broker.get_account().equity)
        except Exception:
            equity = 0.0

        from amms.analysis.portfolio_optimizer import format_allocation, optimize
        try:
            results = optimize(symbols[:20], data, mode=mode)
        except Exception as e:
            return f"optimizer error: {e!r}"

        return format_allocation(results, equity=equity)

    def _momscan(args: list[str]) -> str:
        """Rank the active watchlist by composite momentum score.

        Usage: /momscan [N]  — show top N (default 10)
        Score: 0..100 (RSI reversal + EMA trend + 20d momentum + vol adjustment)
        """
        if data is None:
            return "Data client not wired."

        # Build symbol list from all watchlist sources
        symbols: list[str] = list(static_watchlist)
        if get_wsb_extras is not None:
            symbols += list(get_wsb_extras())
        if db_path is not None:
            from amms.data.dynamic_watchlist import load as load_dyn
            symbols += list(load_dyn(db_path))
        symbols = list(dict.fromkeys(s.upper() for s in symbols))

        if not symbols:
            return "Watchlist is empty. Add tickers with /add SYM."

        top_n = 10
        if args:
            try:
                top_n = max(1, min(int(args[0]), 20))
            except ValueError:
                return "usage: /momscan [N]  (number of top results)"

        from amms.analysis.momentum_scan import scan
        results = scan(symbols, data, top_n=top_n)
        if not results:
            return "No data available for momentum scan."

        lines = [f"Momentum scan — top {len(results)} of {len(symbols)} tickers:"]
        for i, r in enumerate(results, 1):
            trend_sym = {"strong_bull": "📈", "weak_bull": "↗️", "bear": "📉"}.get(r.ema_trend, "↔️")
            lines.append(
                f"  #{i:<2} {r.symbol:<6}  score {r.score:.0f}/100  {trend_sym}  {r.reason}"
            )
        return "\n".join(lines)

    def _regime(_args: list[str]) -> str:
        """Show the current market regime (bull/neutral/bear) from SPY + VIXY analysis."""
        if data is None:
            return "Data client not wired."
        try:
            from amms.analysis.regime import detect_regime
            regime = detect_regime(data)
        except Exception as e:
            return f"regime detection error: {e!r}"

        emoji = {"bull": "📈", "neutral": "↔️", "bear": "📉"}.get(regime.label, "❓")
        lines = [
            f"Market regime: {emoji} {regime.label.upper()}  (confidence {regime.confidence:.0%})",
            f"Reason: {regime.reason}",
            f"Risk multiplier: {regime.risk_multiplier:.2f}×",
        ]
        if regime.spy_vs_sma50 is not None:
            lines.append(f"  SPY vs SMA-50:  {regime.spy_vs_sma50:+.2f}%")
        if regime.spy_vs_sma200 is not None:
            lines.append(f"  SPY vs SMA-200: {regime.spy_vs_sma200:+.2f}%")
        if regime.vixy_1d_pct is not None:
            lines.append(f"  VIXY 1d:        {regime.vixy_1d_pct:+.2f}%")
        return "\n".join(lines)

    def _circuit(args: list[str]) -> str:
        """Show or reset the circuit breaker state.

        The circuit breaker blocks new buys after excessive daily loss or
        consecutive losing trades. Resets automatically at next trading day.

        Usage:
          /circuit         — show current circuit breaker state
          /circuit reset   — manually reset today's circuit (admin)
        """
        if conn is None:
            return "DB not wired."

        from amms.risk.circuit_breaker import (
            load_state,
            reset_circuit as do_reset,
        )

        if args and args[0].lower() == "reset":
            do_reset(conn)
            return "Circuit breaker reset for today. Trading will resume."

        try:
            state = load_state(conn)
        except Exception as e:
            return f"circuit breaker error: {e!r}"

        status = "🔴 TRIPPED — new buys blocked" if state.tripped else "🟢 OK — trading allowed"
        lines = [
            f"Circuit breaker status: {status}",
            f"  Trade date:        {state.trade_date}",
            f"  Daily loss:        ${state.daily_loss:,.2f}",
            f"  Consecutive losses: {state.consec_losses}",
        ]
        if state.tripped:
            lines.append("\nUse /circuit reset to manually unblock (use with caution).")
        return "\n".join(lines)

    def _backhist(args: list[str]) -> str:
        """Show history of saved backtest reports.

        Usage: /backhist [N]  — show last N reports (default 5)
        Reports are saved by /backtest when a report_dir is configured.
        """
        limit = 5
        if args:
            try:
                limit = max(1, min(int(args[0]), 20))
            except ValueError:
                return "usage: /backhist [N]  (e.g. /backhist 10)"

        from amms.backtest.report import format_report_summary, load_report_history
        reports = load_report_history(limit=limit)
        if not reports:
            return "No backtest reports saved yet. Run /backtest to generate one."
        lines = [f"Last {len(reports)} backtest report(s):"]
        for r in reports:
            lines.append(f"  {format_report_summary(r)}")
        return "\n".join(lines)

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
            "/setlist SYM [SYM ...] — bulk-replace the dynamic watchlist\n"
            "/calendar — NYSE market hours + upcoming holidays\n"
            "/heatmap — positions ranked by % gain/loss with ASCII bars\n"
            "/limit [N|off] — show or set max daily buy orders\n"
            "/drawdown — detailed drawdown analytics (current, worst, recovery)\n"
            "/alloc — portfolio sector allocation vs equal-weight target\n"
            "/vol [SYM] — realized volatility + ATR for open positions or a ticker\n"
            "/reload — hot-reload config.yaml without restarting the bot\n"
            "/bench [N] — compare portfolio return vs SPY over N days (default 30)\n"
            "/targets — entry price, stop-loss, and 2R take-profit per position\n"
            "/mdd — 5 worst and 3 best single-day equity moves\n"
            "/optout SYM — block ticker from being traded (still on watchlist)\n"
            "/optout SYM remove — unblock ticker\n"
            "/optout list — show blocked tickers\n"
            "/note SYM [text] — save or read a freetext note for a ticker\n"
            "/note list — list all tickers with notes\n"
            "/recap — brief daily summary (equity, trades, top mover, status)\n"
            "/rsi [SYM] — 14-day RSI for open positions or a ticker\n"
            "/ema [SYM] — EMA-20 / EMA-50 crossover status\n"
            "/macd [SYM] — MACD(12,26,9) line, signal, histogram\n"
            "/score [SYM] — composite signal score (-100..+100) for a ticker\n"
            "/filter [score N|rsi oversold|bull|bear] — screen watchlist by signal\n"
            "/sizing SYM [price] — recommended share quantity given equity + risk\n"
            "/winloss [SYM] — win/loss count and net P&L by ticker\n"
            "/hold — average holding period per ticker from completed trades\n"
            "/backhist [N] — show history of saved backtest reports\n"
            "/circuit — show circuit breaker state (auto-blocks on heavy losses)\n"
            "/circuit reset — manually unblock the circuit breaker\n"
            "/regime — detect current market regime (bull/neutral/bear) from SPY + VIXY\n"
            "/momscan [N] — rank watchlist by composite momentum score (top N)\n"
            "/optimize [equal|momentum|inversevol] — recommended portfolio weights\n"
            "/vwap [SYM] — price vs VWAP deviation for positions or ticker\n"
            "/strategies — list all registered trading strategies\n"
            "/stopopt [SYM] — suggest stop-loss % based on ATR (tight/balanced/wide)\n"
            "/chart — ASCII equity curve sparkline from equity history\n"
            "/risk2r — show open P&L per position in R-units (multiples of initial risk)\n"
            "/rotation — sector rotation: which sectors outperform SPY (20d momentum)\n"
            "/bb [SYM] — Bollinger Bands for open positions or a single ticker\n"
            "/volspike [SYM] — volume spike ratio vs 20-day average\n"
            "/divergence [SYM] — RSI/price divergence signal (bullish/bearish/hidden)\n"
            "/zscore [SYM] — Z-score: how far price is from its 20-bar mean\n"
            "/adx [SYM] — ADX trend strength: ranging vs trending, +DI/-DI direction\n"
            "/stoch [SYM] — Stochastic %%K/%%D: oversold/overbought + crossover signals\n"
            "/confluence [SYM] — multi-indicator confluence score (RSI+MACD+BB+ADX+Stoch)\n"
            "/ichimoku [SYM] — Ichimoku Cloud: cloud position + Tenkan/Kijun momentum\n"
            "/watchdog — daily risk watchdog: circuit, regime, P&L, technical warnings\n"
            "/sar [SYM] — Parabolic SAR: trend direction + stop distance\n"
            "/trend [SYM] — multi-indicator trend summary (SMA/EMA/RSI/MACD/ADX)\n"
            "/obv [SYM] — On-Balance Volume: buying/selling pressure + divergence\n"
            "/attribution — P&L attribution: which positions drive portfolio returns\n"
            "/vwap [SYM] — VWAP with ±1σ/±2σ bands and price deviation\n"
            "/volprofile [SYM] — Volume Profile: Point of Control + 70%% Value Area\n"
            "/forecast [SYM] [DAYS] — statistical price forecast with 68%%/95%% CI bands\n"
            "/swings [SYM] — swing high/low pivots: trend, stop suggestion, breakout flag\n"
            "/aging — position aging: hold time, style, P&L, overstay flags\n"
            "/corrmatrix — portfolio correlation matrix + diversification score\n"
            "/gaps [SYM] — gap analysis: recent price gaps, fill status, gap S/R levels\n"
            "/sectorheat — sector momentum heatmap: 5d/20d/60d ranked by composite score\n"
            "/btstats [DAYS] — extended backtest stats: Calmar, Sortino, recovery, streaks\n"
            "/meanrev [SYM] — mean reversion score: how stretched is price from mean (0-100)\n"
            "/breadth — portfolio breadth: pct positions above VWAP/RSI50/SMA20/OBV\n"
            "/trendlines [SYM] — auto-detect support/resistance trend lines + pattern\n"
            "/roc [SYM] — Rate of Change: 10/20/50-bar momentum + trend alignment\n"
            "/wr [SYM] — Williams %%R: overbought/oversold oscillator (0 to -100)\n"
            "/cci [SYM] — Commodity Channel Index: overbought/oversold (+100/-100)\n"
            "/pairs SYM1 SYM2 — pairs trading: ratio Z-score, correlation, mean-reversion signal\n"
            "/vregime [SYM] — volatility regime: ATR percentile + size multiplier\n"
            "/health [SYM] — full indicator health check for a position\n"
            "/candles [SYM] — candlestick pattern detector (Doji/Hammer/Engulfing/etc.)\n"
            "/fib [SYM] — Fibonacci retracement/extension levels from swing high/low\n"
            "/journalstats — extended trade stats: expectancy, Sharpe, streaks, hold\n"
            "/stress [SCENARIO] — portfolio stress test (2008/covid/dotcom/custom)\n"
            "/strategy — regime-based strategy recommendation\n"
            "/scan2 [min_score] — multi-indicator scanner: BB+RSI+Stoch+MACD setups\n"
            "/montecarlo [N] — Monte Carlo simulation from trade history (1000 paths)\n"
            "/kelly PRICE STOP%% [WIN_RATE] — Kelly criterion + fixed-fraction sizing\n"
            "/rr ENTRY STOP TARGET [QTY] — risk/reward calculator\n"
            "/signals — last 10 strategy signals\n"
            "/lastorders — last 10 orders\n"
            "/scan — run WSB Auto-Discovery now\n"
            "/isin SYM — look up the ISIN for a ticker (debug)\n"
            "/buylist — preview what the bot would buy/sell right now\n"
            "/sentiment [SYM] — WSB mention rank and sentiment for a ticker\n"
            "/profit [day|week|month|all] — realized P&L breakdown by period\n"
            "/upcoming — upcoming earnings dates for held positions (14d)\n"
            "/compare SYM1 SYM2 — side-by-side ticker comparison (price, change, sector)\n"
            "/riskreport — full risk diagnostic (drawdown, sector conc., correlation)\n"
            "/streak — current win/loss streak from completed trade history\n"
            "/sharpe — rolling Sharpe ratio + max drawdown from equity curve\n"
            "/budget — available buying power and position slot breakdown\n"
            "/corr — pairwise return correlation between held positions\n"
            "/journal [SYM] — completed trade pairs (BUY→SELL) with realized P&L\n"
            "/top — best and worst open positions by unrealized P&L %%\n"
            "/news [SYM] — recent news headlines for a ticker (or open positions)\n"
            "/export [N] — export last N filled orders as CSV text\n"
            "/fees [BPS] — estimate simulated transaction cost (default 5 bps)\n"
            "/summary — AI-generated narrative of the current portfolio state\n"
            "/ping — health check (shows timestamp and equity)\n"
            "/version — git sha and branch of the running bot\n"
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
        "budget": _budget,
        "corr": _corr,
        "correlation": _corr,
        "streak": _streak,
        "sharpe": _sharpe,
        "riskreport": _riskreport,
        "upcoming": _upcoming,
        "earnings": _upcoming,  # alias
        "compare": _compare,
        "sentiment": _sentiment,
        "profit": _profit,
        "ping": _ping,
        "version": _version,
        "fees": _fees,
        "export": _export,
        "setlist": _setlist,
        "calendar": _calendar,
        "heatmap": _heatmap,
        "limit": _limit,
        "drawdown": _drawdown,
        "alloc": _alloc,
        "vol": _vol,
        "reload": _reload,
        "bench": _bench,
        "benchmark": _bench,
        "targets": _targets,
        "mdd": _mdd,
        "optout": _optout,
        "block": _optout,
        "note": _note,
        "recap": _recap,
        "rsi": _rsi,
        "ema": _ema_cmd,
        "macd": _macd_cmd,
        "score": _score,
        "filter": _filter,
        "sizing": _sizing,
        "size": _sizing,
        "winloss": _winloss,
        "wl": _winloss,
        "hold": _hold,
        "backhist": _backhist,
        "circuit": _circuit,
        "cb": _circuit,
        "regime": _regime,
        "momscan": _momscan,
        "ms": _momscan,
        "optimize": _optimize,
        "opt": _optimize,
        "strategies": _strategies,
        "strats": _strategies,
        "stopopt": _stopopt,
        "chart": _chart,
        "curve": _chart,
        "risk2r": _risk2r,
        "r": _risk2r,
        "rotation": _rotation,
        "sectors2": _rotation,
        "bb": _bb_cmd,
        "volspike": _volspike,
        "divergence": _divergence,
        "div": _divergence,
        "zscore": _zscore_cmd,
        "z": _zscore_cmd,
        "adx": _adx_cmd,
        "stoch": _stoch_cmd,
        "stochastic": _stoch_cmd,
        "confluence": _confluence_cmd,
        "cf": _confluence_cmd,
        "ichimoku": _ichimoku_cmd,
        "ichi": _ichimoku_cmd,
        "watchdog": _watchdog_cmd,
        "wd": _watchdog_cmd,
        "sar": _sar_cmd,
        "trend": _trend_cmd,
        "obv": _obv_cmd,
        "attribution": _attribution_cmd,
        "attr": _attribution_cmd,
        "kelly": _kelly_cmd,
        "rr": _rr_calc,
        "riskreward": _rr_calc,
        "montecarlo": _montecarlo_cmd,
        "mc": _montecarlo_cmd,
        "scan2": _scan2_cmd,
        "signals": _scan2_cmd,
        "vwap": _vwap_cmd,
        "volprofile": _volprofile_cmd,
        "vp": _volprofile_cmd,
        "forecast": _forecast_cmd,
        "fc": _forecast_cmd,
        "swings": _swings_cmd,
        "swing": _swings_cmd,
        "aging": _aging_cmd,
        "age": _aging_cmd,
        "corrmatrix": _corrmatrix_cmd,
        "cm": _corrmatrix_cmd,
        "gaps": _gaps_cmd,
        "sectorheat": _sectorheat_cmd,
        "sh": _sectorheat_cmd,
        "btstats": _btstats_cmd,
        "meanrev": _meanrev_cmd,
        "mr": _meanrev_cmd,
        "breadth": _breadth_cmd,
        "trendlines": _trendlines_cmd,
        "tl": _trendlines_cmd,
        "roc": _roc_cmd,
        "wr": _wr_cmd,
        "williamsr": _wr_cmd,
        "cci": _cci_cmd,
        "pairs": _pairs_cmd,
        "vregime": _vregime_cmd,
        "volregime": _vregime_cmd,
        "health": _health_cmd,
        "candles": _candles_cmd,
        "patterns": _candles_cmd,
        "fib": _fib_cmd,
        "fibonacci": _fib_cmd,
        "journalstats": _journalstats_cmd,
        "js": _journalstats_cmd,
        "stress": _stress_cmd,
        "stresstest": _stress_cmd,
        "strategy": _strategy_selector_cmd,
        "stratselect": _strategy_selector_cmd,
        "tradequality": _tradequality_cmd,
        "tq": _tradequality_cmd,
        "drawdown": _drawdown_cmd,
        "posdd": _posdd_cmd,
        "pdd": _posdd_cmd,
        "volpct": _volpct_cmd,
        "volrank": _volpct_cmd,
        "momentum": _momentum_cmd,
        "mom": _momentum_cmd,
        "srlevels": _srlevels_cmd,
        "sr": _srlevels_cmd,
        "liquidity": _liquidity_cmd,
        "liq": _liquidity_cmd,
        "regime": _regime_cmd,
        "mregime": _regime_cmd,
        "heat": _heat_cmd,
        "pheat": _heat_cmd,
        "concentration": _concentration_cmd,
        "conc": _concentration_cmd,
        "holdtime": _holdtime_cmd,
        "ht": _holdtime_cmd,
        "stopopt": _stopopt_cmd,
        "stop": _stopopt_cmd,
        "rsrank": _rsrank_cmd,
        "rs": _rsrank_cmd,
        "overnight": _overnight_cmd,
        "gap_risk": _overnight_cmd,
        "trendq": _trendq_cmd,
        "tc": _trendq_cmd,
        "breakout": _breakout_cmd,
        "bo": _breakout_cmd,
        "dashboard": _dashboard_cmd,
        "dash": _dashboard_cmd,
        "watch": _watch_cmd,
        "wl": _watch_cmd,
        "atrtargets": _targets_cmd,
        "atrt": _targets_cmd,
        "kelly": _kelly_cmd,
        "ks": _kelly_cmd,
        "beta": _beta_cmd,
        "pbeta": _beta_cmd,
        "riskratios": _riskratios_cmd,
        "perf": _riskratios_cmd,
        "tstreak": _tstreak_cmd,
        "tradestreak": _tstreak_cmd,
        "vpdetail": _vpdetail_cmd,
        "vpd": _vpdetail_cmd,
        "expectancy": _expectancy_cmd,
        "exp": _expectancy_cmd,
        "dduration": _dduration_cmd,
        "underwater": _dduration_cmd,
        "imom": _imom_cmd,
        "intraday": _imom_cmd,
        "sectordetail": _sectordetail_cmd,
        "sxp": _sectordetail_cmd,
        "risksize": _risksize_cmd,
        "rs2": _risksize_cmd,
        "ecal": _ecal_cmd,
        "ecaladd": _ecal_cmd,
        "volcone": _volcone_cmd,
        "hvcone": _volcone_cmd,
        "jsum": _jsum_cmd,
        "jmonth": _jsum_cmd,
        "screen": _screen_cmd,
        "screener": _screen_cmd,
        "holdreturn": _holdreturn_cmd,
        "dreturn": _holdreturn_cmd,
        "sigagg": _sigagg_cmd,
        "consensus": _sigagg_cmd,
        "timing": _trade_timing_cmd,
        "tradetiming": _trade_timing_cmd,
        "pcal": _pcal_cmd,
        "profitcal": _pcal_cmd,
        "symperf": _symperf_cmd,
        "symbolperf": _symperf_cmd,
        "perftrend": _perftrend_cmd,
        "mtrend": _perftrend_cmd,
        "sizeperf": _sizeperf_cmd,
        "sizevret": _sizeperf_cmd,
        "tfreq": _tfreq_cmd,
        "tradefreq": _tfreq_cmd,
        "scorecard": _scorecard_cmd,
        "grade": _scorecard_cmd,
        "ruin": _ruin_cmd,
        "ruinrisk": _ruin_cmd,
        "paydist": _paydist_cmd,
        "retdist": _paydist_cmd,
        "wrstab": _wrstab_cmd,
        "wrstability": _wrstab_cmd,
        "sectorwr": _sectorwr_cmd,
        "secwr": _sectorwr_cmd,
        "autocorr": _autocorr_cmd,
        "hothand": _autocorr_cmd,
        "mtf": _mtf_cmd,
        "multitf": _mtf_cmd,
        "mstruct": _mstruct_cmd,
        "mktstruct": _mstruct_cmd,
        "squeeze": _squeeze_cmd,
        "compress": _squeeze_cmd,
        "pivots": _pivots_cmd,
        "pp": _pivots_cmd,
        "fvg": _fvg_cmd,
        "imbalance": _fvg_cmd,
        "ribbon": _ribbon_cmd,
        "maribbon": _ribbon_cmd,
        "ha": _heikin_ashi_cmd,
        "heikin": _heikin_ashi_cmd,
        "heikinashi": _heikin_ashi_cmd,
        "avwap": _avwap_cmd,
        "anchored": _avwap_cmd,
        "ofi": _ofi_cmd,
        "orderflow": _ofi_cmd,
        "volskew": _volskew_cmd,
        "vskew": _volskew_cmd,
        "ptc": _ptconsensus_cmd,
        "ptconsensus": _ptconsensus_cmd,
        "regsize": _regime_sizer_cmd,
        "regkelly": _regime_sizer_cmd,
        "tcluster": _tcluster_cmd,
        "tradecluster": _tcluster_cmd,
        "lossav": _lossaversion_cmd,
        "disposition": _lossaversion_cmd,
        "corrb": _corrbreakdown_cmd,
        "corrbreakdown": _corrbreakdown_cmd,
        "cvar": _cvar_cmd,
        "es": _cvar_cmd,
        "gapfill": _gapfill_cmd,
        "fillgap": _gapfill_cmd,
        "exitq": _exitquality_cmd,
        "exitquality": _exitquality_cmd,
        "calanom": _calendaranomaly_cmd,
        "seasonality": _calendaranomaly_cmd,
        "wscore": _wscore_cmd,
        "watchscore": _wscore_cmd,
        "ddcurve": _ddcurve_cmd,
        "ddanalysis": _ddcurve_cmd,
        "streakadv": _streak_cmd,
        "streakanalysis": _streak_cmd,
        "holdtime": _holdtime_cmd,
        "holdanalysis": _holdtime_cmd,
        "tsector": _tradesector_cmd,
        "tradesector": _tradesector_cmd,
        "pfdecomp": _pfdecomp_cmd,
        "profitfactor": _pfdecomp_cmd,
        "mtfmom": _mtfmom_cmd,
        "mtf": _mtfmom_cmd,
        "bbsq": _bbsqueeze_cmd,
        "bbsqueeze": _bbsqueeze_cmd,
        "breadthproxy": _breadthproxy_cmd,
        "mktbreadth": _breadthproxy_cmd,
        "fib": _fib_cmd,
        "fiblevels": _fib_cmd,
        "exhaust": _exhaust_cmd,
        "trendexhaust": _exhaust_cmd,
        "volfcast": _volfcast_cmd,
        "ewmavol": _volfcast_cmd,
        "regtrans": _regtrans_cmd,
        "regimetrans": _regtrans_cmd,
        "sigstr": _sigstrength_cmd,
        "signalstrength": _sigstrength_cmd,
        "mrband": _mrband_cmd,
        "meanrevband": _mrband_cmd,
        "candle": _candle_cmd,
        "candlepattern": _candle_cmd,
        "keltner": _keltner_cmd,
        "kc": _keltner_cmd,
        "supertrend": _supertrend_cmd,
        "st": _supertrend_cmd,
        "donchian": _donchian_cmd,
        "dc": _donchian_cmd,
        "cmf": _cmf_cmd,
        "chaikin": _cmf_cmd,
        "psar": _psar_cmd,
        "parabolicsar": _psar_cmd,
        "pascore": _pascore_cmd,
        "priceaction": _pascore_cmd,
        "swingpts": _swingpts_cmd,
        "swinglevels": _swingpts_cmd,
    }
