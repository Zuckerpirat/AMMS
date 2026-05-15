"""Volatility regime detector.

Classifies current market volatility as low/normal/high/extreme by
comparing the current ATR to its historical distribution (percentile ranking).

This is different from a simple ATR threshold — it adapts to the
symbol's own historical volatility behavior.

Regime classification:
  ATR percentile < 25th  → low (favorable for trending strategies)
  ATR percentile 25-75th → normal (standard conditions)
  ATR percentile 75-90th → high (wide stops needed, reduce size)
  ATR percentile > 90th  → extreme (use minimal size or stand aside)
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class VolatilityRegime:
    symbol: str
    current_atr: float
    atr_pct_of_price: float   # ATR as % of current price
    percentile: float          # 0..100 — where current ATR falls in history
    regime: str                # "low" | "normal" | "high" | "extreme"
    recommended_size_mult: float  # suggested position size multiplier (0.25..1.0)
    bars_used: int


def classify(bars: list[Bar], atr_period: int = 14) -> VolatilityRegime | None:
    """Classify volatility regime for the given bar series.

    Needs at least atr_period × 3 bars for meaningful percentile ranking.
    """
    symbol = bars[0].symbol if bars else "?"
    if len(bars) < atr_period * 2 + 1:
        return None

    from amms.features.volatility import atr as compute_atr

    # Compute rolling ATR for each window
    atr_history: list[float] = []
    for i in range(atr_period, len(bars)):
        window = bars[i - atr_period: i + 1]
        val = compute_atr(window, atr_period)
        if val is not None:
            atr_history.append(val)

    if not atr_history:
        return None

    current_atr = atr_history[-1]
    price = bars[-1].close
    atr_pct = (current_atr / price * 100) if price > 0 else 0.0

    # Compute percentile rank (use strict < to avoid 100% for equal ATRs)
    history = atr_history[:-1]
    n_below = sum(1 for v in history if v < current_atr)
    n_equal = sum(1 for v in history if v == current_atr)
    percentile = (n_below + 0.5 * n_equal) / max(len(history), 1) * 100

    if percentile < 25:
        regime = "low"
        size_mult = 1.0
    elif percentile < 75:
        regime = "normal"
        size_mult = 0.9
    elif percentile < 90:
        regime = "high"
        size_mult = 0.6
    else:
        regime = "extreme"
        size_mult = 0.25

    return VolatilityRegime(
        symbol=symbol,
        current_atr=round(current_atr, 4),
        atr_pct_of_price=round(atr_pct, 2),
        percentile=round(percentile, 1),
        regime=regime,
        recommended_size_mult=size_mult,
        bars_used=len(bars),
    )
