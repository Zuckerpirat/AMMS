"""Macro market-regime indicators.

A first, intentionally small step toward CLAUDE.md's "macro and
geopolitical awareness" goal. Currently exposes a single binary
"is the market stressed?" signal derived from VIXY ETF moves —
when the volatility complex spikes hard, the bot should hesitate
to add new long exposure.

Why VIXY and not ^VIX:
  ^VIX is a non-tradable index that Alpaca's stocks API doesn't return.
  VIXY is the iPath VIX short-term futures ETF and tracks it closely
  enough to use as a stress proxy. Available through the same
  snapshots endpoint we already use for the watchlist.

Output contract:
  MacroRegime(level: "calm" | "elevated" | "stressed", reason: str,
              vixy_1d_pct: float, vixy_1w_pct: float)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds — chosen conservatively. A 5% intraday VIXY move is large;
# 15% over a week implies a sustained volatility regime shift.
DEFAULT_DAY_PCT_STRESS = 5.0
DEFAULT_WEEK_PCT_STRESS = 15.0
DEFAULT_DAY_PCT_ELEVATED = 2.5


@dataclass(frozen=True)
class MacroRegime:
    level: str  # "calm" | "elevated" | "stressed"
    reason: str
    vixy_1d_pct: float
    vixy_1w_pct: float

    @property
    def is_stressed(self) -> bool:
        return self.level == "stressed"


def _regime_from_pcts(
    day: float, week: float, *,
    day_pct_stress: float, week_pct_stress: float, day_pct_elevated: float,
    source: str = "VIXY",
) -> MacroRegime:
    """Classify regime from pre-computed percentage changes."""
    if day >= day_pct_stress or week >= week_pct_stress:
        reason = f"{source} 1d {day:+.1f}% / 1w {week:+.1f}% — high volatility regime"
        return MacroRegime("stressed", reason, day, week)
    if day >= day_pct_elevated:
        reason = f"{source} 1d {day:+.1f}% — elevated volatility"
        return MacroRegime("elevated", reason, day, week)
    return MacroRegime(
        "calm",
        f"{source} 1d {day:+.1f}% / 1w {week:+.1f}% — quiet",
        day,
        week,
    )


def _compute_regime_from_bars(
    data, symbol: str, *,
    day_pct_stress: float, week_pct_stress: float, day_pct_elevated: float,
) -> MacroRegime | None:
    """Fallback: compute regime from recent bar data instead of snapshots.

    Returns None if insufficient bars are available.
    """
    try:
        bars = data.get_bars(symbol, limit=10)
    except Exception:
        return None
    if not bars or len(bars) < 2:
        return None
    try:
        last_close = float(bars[-1].close)
        prev_close = float(bars[-2].close)
        day_pct = (last_close / prev_close - 1.0) * 100.0 if prev_close > 0 else 0.0
        # Week approximation: close 5 bars ago
        week_bar_close = float(bars[max(0, len(bars) - 6)].close)
        week_pct = (last_close / week_bar_close - 1.0) * 100.0 if week_bar_close > 0 else 0.0
        return _regime_from_pcts(
            day_pct, week_pct,
            day_pct_stress=day_pct_stress,
            week_pct_stress=week_pct_stress,
            day_pct_elevated=day_pct_elevated,
            source=f"{symbol}(bars)",
        )
    except Exception:
        return None


def compute_regime(
    data,
    *,
    symbol: str = "VIXY",
    day_pct_stress: float = DEFAULT_DAY_PCT_STRESS,
    week_pct_stress: float = DEFAULT_WEEK_PCT_STRESS,
    day_pct_elevated: float = DEFAULT_DAY_PCT_ELEVATED,
) -> MacroRegime:
    """Fetch VIXY snapshot and classify the regime.

    Primary method: uses get_snapshots() for intraday precision.
    Fallback: uses get_bars() if snapshots unavailable (test environments,
    brokers without snapshot support).

    Falls back to a neutral 'calm' if both methods fail so a
    market-data hiccup doesn't accidentally pause the bot.
    """
    snapshot_raised = False
    snap: dict = {}
    try:
        snap = data.get_snapshots([symbol]) or {}
    except Exception:
        logger.warning("macro regime snapshot failed", exc_info=True)
        snapshot_raised = True

    entry = snap.get(symbol) or {}
    day = float(entry.get("change_pct") or 0.0)
    week = float(entry.get("change_pct_week") or 0.0)

    # If snapshot returned real data, use it
    if entry:
        return _regime_from_pcts(
            day, week,
            day_pct_stress=day_pct_stress,
            week_pct_stress=week_pct_stress,
            day_pct_elevated=day_pct_elevated,
        )

    # Fallback to bar-based computation — only when snapshot call worked but
    # returned no entry (broker doesn't support this ticker via snapshots).
    # When the call itself raised, the client likely doesn't support snapshots
    # at all, so bar fallback would just add noise.
    if not snapshot_raised:
        bar_regime = _compute_regime_from_bars(
            data, symbol,
            day_pct_stress=day_pct_stress,
            week_pct_stress=week_pct_stress,
            day_pct_elevated=day_pct_elevated,
        )
        if bar_regime is not None:
            logger.debug("macro regime computed from bars (snapshot returned empty)")
            return bar_regime

    return MacroRegime("calm", "VIXY data unavailable — assuming calm", 0.0, 0.0)
