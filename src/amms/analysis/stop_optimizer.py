"""Stop-loss optimizer.

Given historical bar data for a symbol, suggests a stop-loss percentage
based on the historical Average True Range and maximum intraday swings.
Helps calibrate tight vs loose stops without data fitting.

Strategy:
  - Compute ATR-14 as a percentage of price (ATR%)
  - Suggest: 1.5× ATR% as a balanced stop (catches most noise, cuts big moves)
  - Also report: 1× ATR% (tight), 2× ATR% (wide), max 20d daily range
  - No overfitting — based solely on volatility, not specific price targets.

Lives in the analysis layer: pure computation, no trade decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amms.data.bars import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopSuggestion:
    symbol: str
    current_price: float
    atr_pct: float | None          # ATR-14 as % of price
    stop_tight_pct: float | None   # 1× ATR%
    stop_balanced_pct: float | None # 1.5× ATR%
    stop_wide_pct: float | None    # 2× ATR%
    max_daily_range_pct: float | None  # max (high-low)/close over 20d
    recommendation: str


def suggest_stop(symbol: str, bars: list[Bar]) -> StopSuggestion:
    """Compute stop-loss suggestions for a single symbol."""
    if not bars:
        return StopSuggestion(
            symbol=symbol, current_price=0.0,
            atr_pct=None, stop_tight_pct=None,
            stop_balanced_pct=None, stop_wide_pct=None,
            max_daily_range_pct=None,
            recommendation="No bar data available.",
        )

    from amms.features.volatility import atr as compute_atr

    price = bars[-1].close
    atr_val = compute_atr(bars, 14)

    atr_pct = (atr_val / price * 100) if (atr_val and price > 0) else None

    # Daily range % (max over last 20 bars)
    daily_ranges = []
    for b in bars[-20:]:
        if b.close > 0:
            daily_ranges.append((b.high - b.low) / b.close * 100)
    max_range = max(daily_ranges) if daily_ranges else None

    if atr_pct is not None:
        tight = round(atr_pct, 2)
        balanced = round(atr_pct * 1.5, 2)
        wide = round(atr_pct * 2.0, 2)
        if atr_pct < 1.0:
            rec = f"Low volatility: tight {tight:.1f}% stop may work."
        elif atr_pct < 2.5:
            rec = f"Moderate volatility: balanced {balanced:.1f}% stop recommended."
        else:
            rec = f"High volatility: wide {wide:.1f}% stop or reduce position."
    else:
        tight = balanced = wide = None
        rec = "Insufficient bar history for ATR calculation."

    return StopSuggestion(
        symbol=symbol,
        current_price=price,
        atr_pct=atr_pct,
        stop_tight_pct=tight,
        stop_balanced_pct=balanced,
        stop_wide_pct=wide,
        max_daily_range_pct=max_range,
        recommendation=rec,
    )
