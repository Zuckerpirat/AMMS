"""Background scheduler — runs the Auto-Trader on a watchlist on a clock.

Thread-based; safe to start/stop from bot commands. Each tick:

  1. (optional) Check market hours via Alpaca clock
  2. Run AutoTrader.run_watchlist(symbols)
  3. Log results to a journal file
  4. Sleep until next tick

The scheduler is single-instance (singleton container) and survives
across bot commands but not process restarts. State (tick count, last
results) is exposed via `.status()`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStatus:
    running: bool
    tick_seconds: int
    tick_count: int
    started_at: str        # ISO timestamp
    last_tick_at: str
    last_tick_summary: str
    symbols: list[str]
    market_hours_only: bool


class TraderScheduler:
    """Background loop that drives the Auto-Trader."""

    def __init__(self, auto_trader, symbols: list[str], *,
                 tick_seconds: int = 300,
                 market_hours_only: bool = False,
                 clock_fn = None,
                 journal_path: Path | None = None):
        self.auto_trader = auto_trader
        self.symbols = [s.upper() for s in symbols]
        self.tick_seconds = max(10, int(tick_seconds))
        self.market_hours_only = market_hours_only
        self.clock_fn = clock_fn   # optional callable returning ClockStatus-like obj
        self.journal_path = journal_path or Path("scheduler_journal.log")

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.tick_count = 0
        self.started_at = ""
        self.last_tick_at = ""
        self.last_tick_summary = ""

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
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lock:
            t = self._thread
            if t is None or not t.is_alive():
                return False
            self._stop_event.set()
        t.join(timeout=timeout)
        logger.info("Scheduler stopped")
        return True

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_symbols(self, symbols: list[str]) -> None:
        with self._lock:
            self.symbols = [s.upper() for s in symbols]

    def set_tick_seconds(self, seconds: int) -> None:
        with self._lock:
            self.tick_seconds = max(10, int(seconds))

    # ── Core loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                logger.exception("Scheduler tick error: %s", exc)
            # Sleep in small slices so stop() is responsive
            for _ in range(self.tick_seconds):
                if self._stop_event.is_set():
                    return
                time.sleep(1.0)

    def _tick_once(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Market-hours gate
        if self.market_hours_only and self.clock_fn is not None:
            try:
                clock = self.clock_fn()
                if not getattr(clock, "is_open", True):
                    self.last_tick_at = now_iso
                    self.last_tick_summary = "market closed — skipped"
                    self._append_journal(f"{now_iso} market closed\n")
                    return
            except Exception as exc:
                logger.warning("Clock check failed: %s", exc)

        with self._lock:
            syms = list(self.symbols)

        if not syms:
            self.last_tick_at = now_iso
            self.last_tick_summary = "no symbols configured"
            return

        results = self.auto_trader.run_watchlist(syms)
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
