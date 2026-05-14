"""Drawdown monitor.

Watches account equity against its recent peak and emits an alert when
the bot has lost more than the configured percentage. Lives in the risk
layer per CLAUDE.md: drawdown protection is a risk-layer concern.

Stateless helper; the scheduler keeps the per-day mute flag so we don't
spam Telegram if equity hovers around the threshold.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


DEFAULT_DRAWDOWN_ALERT_PCT = 5.0


@dataclass(frozen=True)
class DrawdownState:
    peak_equity: float
    current_equity: float
    drawdown_pct: float  # negative number, e.g. -7.5 means 7.5% below peak

    @property
    def is_breached(self) -> bool:
        return self.drawdown_pct < 0


def compute_drawdown(
    conn: sqlite3.Connection, current_equity: float, lookback_days: int = 30
) -> DrawdownState:
    """Compare current equity to the highest equity_snapshot in the
    lookback window. Returns drawdown_pct (negative when below peak)."""
    row = conn.execute(
        "SELECT MAX(equity) AS peak FROM equity_snapshots "
        "WHERE substr(ts, 1, 10) >= date('now', ?)",
        (f"-{lookback_days} day",),
    ).fetchone()
    peak_raw = row[0] if row else None
    peak = float(peak_raw) if peak_raw else current_equity
    peak = max(peak, current_equity)
    if peak <= 0:
        return DrawdownState(peak, current_equity, 0.0)
    pct = (current_equity - peak) / peak * 100
    return DrawdownState(peak, current_equity, pct)


def should_alert(
    state: DrawdownState,
    *,
    threshold_pct: float = DEFAULT_DRAWDOWN_ALERT_PCT,
) -> bool:
    """True when the current drawdown exceeds the alert threshold.

    ``threshold_pct`` is expressed as a positive number (e.g. 5.0 =
    alert at 5% below peak).
    """
    return state.drawdown_pct <= -abs(threshold_pct)
