"""Background scheduler — runs the Auto-Trader on a watchlist on a clock.

Thread-based; safe to start/stop from bot commands. Each tick:

  1. (optional) Check market hours via Alpaca clock
  2. Run AutoTrader.run_watchlist(symbols)
  3. Notify via Telegram for every trade, stop-loss, or risk event
  4. Log results to a journal file
  5. Sleep until next tick

Proactive notifications sent automatically:
  - Every bought/sold/closed position (with reason, price, P&L)
  - ATR stop-loss triggers
  - Risk guard events (drawdown alert, killswitch)
  - Morning briefing at first market-open tick
  - End-of-day summary at last tick before market close
  - Auto-scanner discoveries (new watchlist symbols added)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_MARKET_OPEN_HOUR = 9     # ET — approximate, used only if no clock_fn
_MARKET_CLOSE_HOUR = 16


@dataclass
class SchedulerStatus:
    running: bool
    tick_seconds: int
    tick_count: int
    started_at: str
    last_tick_at: str
    last_tick_summary: str
    symbols: list[str]
    market_hours_only: bool


class TraderScheduler:
    """Background loop that drives the Auto-Trader with proactive Telegram notifications."""

    def __init__(self, auto_trader, symbols: list[str], *,
                 tick_seconds: int = 300,
                 market_hours_only: bool = False,
                 clock_fn=None,
                 journal_path: Path | None = None,
                 db_conn=None,
                 risk_guard=None,
                 notifier=None,
                 auto_scanner=None):
        self.auto_trader = auto_trader
        self.symbols = [s.upper() for s in symbols]
        self.tick_seconds = max(10, int(tick_seconds))
        self.market_hours_only = market_hours_only
        self.clock_fn = clock_fn
        self.journal_path = journal_path or Path("scheduler_journal.log")
        self.db_conn = db_conn
        self.risk_guard = risk_guard
        self.notifier = notifier       # Notifier | NullNotifier
        self.auto_scanner = auto_scanner  # optional AutoScanner instance

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.tick_count = 0
        self.started_at = ""
        self.last_tick_at = ""
        self.last_tick_summary = ""

        # State for one-per-day notifications
        self._morning_sent_date: str = ""
        self._eod_sent_date: str = ""
        self._last_drawdown_alert_date: str = ""

    # ── Control ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self.started_at = datetime.now(timezone.utc).isoformat()
            self._thread = threading.Thread(
                target=self._run_loop, name="amms-scheduler", daemon=True
            )
            self._thread.start()
        logger.info("Scheduler started (%d symbols, every %ds)", len(self.symbols), self.tick_seconds)
        self._notify(
            f"🟢 AMMS Scheduler gestartet\n"
            f"  Symbole: {len(self.symbols)} ({', '.join(self.symbols[:5])}"
            f"{'...' if len(self.symbols) > 5 else ''})\n"
            f"  Intervall: {self.tick_seconds}s\n"
            f"  Marktzeiten: {'nur offen' if self.market_hours_only else 'immer'}"
        )
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lock:
            t = self._thread
            if t is None or not t.is_alive():
                return False
            self._stop_event.set()
        t.join(timeout=timeout)
        logger.info("Scheduler stopped")
        self._notify("🔴 AMMS Scheduler gestoppt.")
        return True

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_symbols(self, symbols: list[str]) -> None:
        with self._lock:
            self.symbols = [s.upper() for s in symbols]

    def set_tick_seconds(self, seconds: int) -> None:
        with self._lock:
            self.tick_seconds = max(10, int(seconds))

    # ── Notification helper ────────────────────────────────────────────────

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier.send(text)
            except Exception as exc:
                logger.debug("Notifier send failed: %s", exc)

    # ── Core loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                logger.exception("Scheduler tick error: %s", exc)
                self._notify(f"⚠️ Scheduler-Fehler: {exc}")
            for _ in range(self.tick_seconds):
                if self._stop_event.is_set():
                    return
                time.sleep(1.0)

    def _tick_once(self) -> None:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        today = now.date().isoformat()

        # Market-hours gate
        is_open = True
        if self.market_hours_only and self.clock_fn is not None:
            try:
                clock = self.clock_fn()
                is_open = getattr(clock, "is_open", True)
                if not is_open:
                    self.last_tick_at = now_iso
                    self.last_tick_summary = "market closed — skipped"
                    self._append_journal(f"{now_iso} market closed\n")
                    return
            except Exception as exc:
                logger.warning("Clock check failed: %s", exc)

        # Morning briefing — once per day at first open tick
        if is_open and today != self._morning_sent_date:
            self._morning_sent_date = today
            self._send_morning_briefing()

        with self._lock:
            syms = list(self.symbols)

        if not syms:
            self.last_tick_at = now_iso
            self.last_tick_summary = "no symbols configured"
            return

        # Auto-scanner: discover new symbols and remove decayed ones (runs hourly)
        if self.auto_scanner is not None:
            try:
                new_syms = self.auto_scanner.scan_and_update(syms)
                if new_syms:
                    with self._lock:
                        for s in new_syms:
                            if s not in self.symbols:
                                self.symbols.append(s)
                        syms = list(self.symbols)
                    self._notify(
                        f"🔍 Auto-Scanner: {len(new_syms)} neue Symbole entdeckt\n"
                        + "\n".join(f"  + {s}" for s in new_syms[:8])
                    )

                # Remove symbols that have decayed (no signal for decay_ticks scans)
                stale = self.auto_scanner.symbols_to_remove()
                if stale:
                    with self._lock:
                        for s in stale:
                            if s in self.symbols:
                                self.symbols.remove(s)
                        syms = list(self.symbols)
                    self._notify(
                        f"⏳ Auto-Scanner: {len(stale)} Symbole entfernt (kein Signal)\n"
                        + "\n".join(f"  - {s}" for s in stale[:8])
                    )
            except Exception as exc:
                logger.debug("Auto-scanner error: %s", exc)

        # Risk guard check before running trades
        if self.risk_guard is not None:
            try:
                self._check_risk_guard(today)
            except Exception as exc:
                logger.debug("Risk guard check failed: %s", exc)

        # Run trading tick
        results = self.auto_trader.run_watchlist(syms)

        # Notify on trades and stops
        self._notify_results(results)

        bought = sum(1 for r in results if r.action == "bought")
        closed = sum(1 for r in results if r.action == "closed")
        skipped = sum(1 for r in results if r.action == "skipped")

        self.tick_count += 1
        self.last_tick_at = now_iso
        self.last_tick_summary = (
            f"{len(results)} symbols, {bought} bought, {closed} closed, {skipped} skipped"
        )
        line = f"{now_iso} tick#{self.tick_count} {self.last_tick_summary}\n"
        for r in results:
            if r.action in {"bought", "closed", "sold"}:
                line += (
                    f"    {r.action.upper():<7} {r.symbol} qty={r.qty:.4f} "
                    f"@ ${r.price:.2f}  score={r.score:+.0f} conf={r.confidence:.0%}\n"
                )
        self._append_journal(line)

        # Update RiskGuard peak equity
        if self.risk_guard is not None:
            try:
                self.risk_guard.update_peak()
            except Exception as exc:
                logger.debug("RiskGuard peak update failed: %s", exc)

        # Record equity snapshot
        if self.db_conn is not None:
            try:
                from amms.data.equity_history import record_snapshot
                record_snapshot(self.db_conn, self.auto_trader.trader)
            except Exception as exc:
                logger.debug("Equity snapshot failed: %s", exc)

        # End-of-day summary — once per day, only after trades ran
        if today != self._eod_sent_date:
            is_near_close = now.hour >= 15 and now.minute >= 45  # 15:45 ET ~ 19:45 UTC
            if is_near_close:
                self._eod_sent_date = today
                self._send_eod_summary()

    # ── Notifications ─────────────────────────────────────────────────────

    def _notify_results(self, results) -> None:
        """Send Telegram message for each trade execution and stop."""
        if self.notifier is None:
            return

        trader = self.auto_trader.trader
        snap = trader.snapshot()
        equity = snap.portfolio_value

        for r in results:
            if r.action == "bought":
                cost = r.qty * r.price
                self._notify(
                    f"📈 GEKAUFT: {r.symbol}\n"
                    f"  Menge: {r.qty:.4f} Stk @ ${r.price:.2f}\n"
                    f"  Kosten: ${cost:.2f}  (DE-Score: {r.score:+.0f}, Konfidenz: {r.confidence:.0%})\n"
                    f"  Grund: {r.reason[:120] if r.reason else '—'}\n"
                    f"  Portfolio: ${equity:,.2f}"
                )

            elif r.action in {"closed", "sold"}:
                # Calculate P&L if position existed
                pnl_str = ""
                try:
                    hist = trader.trade_history
                    # Find most recent buy for this symbol
                    buys = [t for t in hist if t.symbol == r.symbol and t.side == "buy"]
                    if buys:
                        avg_cost = buys[-1].price
                        pnl = (r.price - avg_cost) * r.qty
                        pnl_pct = (r.price / avg_cost - 1.0) * 100.0 if avg_cost > 0 else 0.0
                        sign = "+" if pnl >= 0 else ""
                        pnl_str = f"\n  P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
                except Exception:
                    pass

                is_stop = "stop" in (r.reason or "").lower()
                icon = "🛑" if is_stop else "📉"
                action_word = "STOP-LOSS" if is_stop else "VERKAUFT"
                self._notify(
                    f"{icon} {action_word}: {r.symbol}\n"
                    f"  Menge: {r.qty:.4f} Stk @ ${r.price:.2f}{pnl_str}\n"
                    f"  Grund: {r.reason[:120] if r.reason else '—'}\n"
                    f"  Portfolio: ${equity:,.2f}"
                )

    def _check_risk_guard(self, today: str) -> None:
        """Send drawdown alert if threshold breached (max once per day)."""
        rg = self.risk_guard
        if rg is None or self.notifier is None:
            return

        state = rg.state
        if state.killswitch_armed:
            return  # already reported

        try:
            peak = state.peak_equity
            equity = self.auto_trader.trader.snapshot().portfolio_value
            if peak > 0:
                drawdown_pct = (peak - equity) / peak * 100.0
                threshold = 5.0  # alert at 5% drawdown from peak
                if drawdown_pct >= threshold and today != self._last_drawdown_alert_date:
                    self._last_drawdown_alert_date = today
                    self._notify(
                        f"⚠️ DRAWDOWN-ALERT: {drawdown_pct:.1f}% unter Peak\n"
                        f"  Aktuell: ${equity:,.2f}  |  Peak: ${peak:,.2f}\n"
                        f"  Neue Käufe werden eingeschränkt."
                    )
        except Exception:
            pass

    def _send_morning_briefing(self) -> None:
        """Morning briefing: portfolio status + macro + top signals."""
        if self.notifier is None:
            return
        try:
            trader = self.auto_trader.trader
            snap = trader.snapshot()
            n_pos = len(snap.positions)
            equity = snap.portfolio_value
            cash = snap.cash

            lines = [
                f"🌅 Guten Morgen — Markt öffnet jetzt",
                f"  Portfolio: ${equity:,.2f}  |  Cash: ${cash:,.2f}",
                f"  Offene Positionen: {n_pos}",
            ]

            # Macro regime
            try:
                from amms.data.macro import compute_regime
                regime = compute_regime(self.auto_trader.data)
                icon = "🔴" if regime.is_stressed else ("🟡" if regime.level == "elevated" else "🟢")
                lines.append(f"  Makro: {icon} {regime.level.upper()} — {regime.reason[:80]}")
            except Exception:
                pass

            # Positions snapshot
            if snap.positions:
                lines.append("  Positionen:")
                for sym, pos in list(snap.positions.items())[:5]:
                    pnl_pct = pos.get("unrealized_pnl_pct", 0.0) if isinstance(pos, dict) else 0.0
                    sign = "+" if pnl_pct >= 0 else ""
                    lines.append(f"    {sym}: {sign}{pnl_pct:.1f}%")

            with self._lock:
                syms = list(self.symbols)
            lines.append(f"  Watchlist: {len(syms)} Symbole")
            self._notify("\n".join(lines))
        except Exception as exc:
            logger.debug("Morning briefing failed: %s", exc)

    def _send_eod_summary(self) -> None:
        """End-of-day summary: P&L, trade count, equity curve."""
        if self.notifier is None:
            return
        try:
            trader = self.auto_trader.trader
            snap = trader.snapshot()

            # Count today's trades
            from datetime import date
            today_str = date.today().isoformat()
            today_trades = []
            try:
                hist = trader.trade_history
                today_trades = [t for t in hist if str(t.ts)[:10] == today_str]
            except Exception:
                pass

            buys = sum(1 for t in today_trades if t.side == "buy")
            sells = sum(1 for t in today_trades if t.side == "sell")
            realized = snap.realized_pnl

            lines = [
                f"🌆 Tagesabschluss",
                f"  Portfolio: ${snap.portfolio_value:,.2f}",
                f"  Cash: ${snap.cash:,.2f}",
                f"  Offene Pos.: {len(snap.positions)}",
                f"  Heute: {buys} Käufe, {sells} Verkäufe",
                f"  Realisiert (gesamt): ${realized:+,.2f}",
                f"  Rendite: {snap.total_return_pct:+.2f}%",
            ]
            self._notify("\n".join(lines))
        except Exception as exc:
            logger.debug("EOD summary failed: %s", exc)

    # ── Journal ───────────────────────────────────────────────────────────

    def _append_journal(self, line: str) -> None:
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a") as f:
                f.write(line)
        except Exception as exc:
            logger.warning("Could not write scheduler journal: %s", exc)

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> SchedulerStatus:
        return SchedulerStatus(
            running=self.is_running(),
            tick_seconds=self.tick_seconds,
            tick_count=self.tick_count,
            started_at=self.started_at,
            last_tick_at=self.last_tick_at,
            last_tick_summary=self.last_tick_summary,
            symbols=list(self.symbols),
            market_hours_only=self.market_hours_only,
        )
