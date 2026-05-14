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


def compute_regime(
    data,
    *,
    symbol: str = "VIXY",
    day_pct_stress: float = DEFAULT_DAY_PCT_STRESS,
    week_pct_stress: float = DEFAULT_WEEK_PCT_STRESS,
    day_pct_elevated: float = DEFAULT_DAY_PCT_ELEVATED,
) -> MacroRegime:
    """Fetch VIXY snapshot and classify the regime.

    Falls back to a neutral 'calm' if the snapshot call fails so a
    market-data hiccup doesn't accidentally pause the bot.
    """
    try:
        snap = data.get_snapshots([symbol]) or {}
    except Exception:
        logger.warning("macro regime snapshot failed", exc_info=True)
        return MacroRegime("calm", "VIXY snapshot unavailable", 0.0, 0.0)

    entry = snap.get(symbol) or {}
    day = float(entry.get("change_pct") or 0.0)
    week = float(entry.get("change_pct_week") or 0.0)

    if day >= day_pct_stress or week >= week_pct_stress:
        reason = (
            f"VIXY 1d {day:+.1f}% / 1w {week:+.1f}% — high volatility regime"
        )
        return MacroRegime("stressed", reason, day, week)
    if day >= day_pct_elevated:
        reason = f"VIXY 1d {day:+.1f}% — elevated volatility"
        return MacroRegime("elevated", reason, day, week)
    return MacroRegime(
        "calm",
        f"VIXY 1d {day:+.1f}% / 1w {week:+.1f}% — quiet",
        day,
        week,
    )
