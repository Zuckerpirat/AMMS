"""Williams Fractal Indicator.

Developed by Bill Williams. Fractals identify potential turning points by
finding local extremes in a 5-bar window:

  Up Fractal (bearish):  Middle bar has the highest high of 5 consecutive bars
                         (higher than 2 bars on each side)
  Down Fractal (bullish): Middle bar has the lowest low of 5 consecutive bars

Williams also defined the Alligator (3 SMMAed lines):
  Jaw (Blue):   SMMA(13), offset 8 bars forward
  Teeth (Red):  SMMA(8),  offset 5 bars forward
  Lips (Green): SMMA(5),  offset 3 bars forward

Price above all 3 → uptrend; below all → downtrend; crossing → flat

This module:
  1. Detects recent fractals in the price history
  2. Computes current Alligator state
  3. Provides fractal-based support/resistance levels
  4. Combines into a signal
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FractalPoint:
    bar_index: int      # position in the bars list (from end, 0=latest)
    price: float        # high (up fractal) or low (down fractal)
    fractal_type: str   # "up" or "down"


@dataclass(frozen=True)
class WilliamsFractalReport:
    symbol: str

    # Recent fractals (up to 5 most recent of each type)
    up_fractals: list[FractalPoint]    # bearish fractals (resistance)
    down_fractals: list[FractalPoint]  # bullish fractals (support)

    # Nearest levels
    nearest_resistance: float | None   # nearest up fractal above price
    nearest_support: float | None      # nearest down fractal below price

    # Distance to support/resistance
    resistance_pct: float | None
    support_pct: float | None

    # Alligator
    jaw: float | None        # 13-bar SMMA
    teeth: float | None      # 8-bar SMMA
    lips: float | None       # 5-bar SMMA
    alligator_open: bool     # lips, teeth, jaw separated (trending)
    alligator_sleeping: bool # lines tangled/crossed (consolidation)

    # Trend from Alligator
    alligator_bullish: bool | None   # price above all three
    alligator_bearish: bool | None   # price below all three

    # Score and signal
    score: float
    signal: str   # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    current_price: float
    bars_used: int
    verdict: str


def _smma(closes: list[float], period: int) -> list[float]:
    """Smoothed Moving Average (Wilder's MA)."""
    if len(closes) < period:
        return []
    result = [sum(closes[:period]) / period]
    for c in closes[period:]:
        result.append((result[-1] * (period - 1) + c) / period)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    fractal_window: int = 5,    # bars on each side (total = 2*window+1)
    max_fractals: int = 5,      # max recent fractals to report
) -> WilliamsFractalReport | None:
    """Detect Williams Fractals and compute the Alligator.

    bars: bar objects with .high, .low, .close attributes.
    """
    min_bars = 40  # enough for Alligator (13-bar SMMA) + fractals
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)
    w = fractal_window

    # Fractal detection
    up_fractals_raw:   list[FractalPoint] = []
    down_fractals_raw: list[FractalPoint] = []

    for i in range(w, n - w):
        # Up fractal: high[i] > all surrounding highs
        if all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, w + 1)):
            up_fractals_raw.append(FractalPoint(
                bar_index=n - 1 - i, price=highs[i], fractal_type="up"
            ))

        # Down fractal: low[i] < all surrounding lows
        if all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, w + 1)):
            down_fractals_raw.append(FractalPoint(
                bar_index=n - 1 - i, price=lows[i], fractal_type="down"
            ))

    # Sort by recency (low bar_index = recent)
    up_fractals_raw.sort(key=lambda f: f.bar_index)
    down_fractals_raw.sort(key=lambda f: f.bar_index)

    up_fractals   = up_fractals_raw[:max_fractals]
    down_fractals = down_fractals_raw[:max_fractals]

    # Nearest resistance (up fractal above current price)
    resistances = [f.price for f in up_fractals if f.price > current]
    nearest_res = min(resistances) if resistances else None
    res_pct = (nearest_res - current) / current * 100.0 if nearest_res else None

    # Nearest support (down fractal below current price)
    supports = [f.price for f in down_fractals if f.price < current]
    nearest_sup = max(supports) if supports else None
    sup_pct = (current - nearest_sup) / current * 100.0 if nearest_sup else None

    # Alligator (using current bar SMMA, not offset)
    jaw_vals   = _smma(closes, 13)
    teeth_vals = _smma(closes, 8)
    lips_vals  = _smma(closes, 5)

    jaw   = jaw_vals[-1]   if jaw_vals   else None
    teeth = teeth_vals[-1] if teeth_vals else None
    lips  = lips_vals[-1]  if lips_vals  else None

    # Alligator state
    if jaw is not None and teeth is not None and lips is not None:
        # Open: lines in order (lips > teeth > jaw for bull, opposite for bear)
        alligator_bull_open = lips > teeth > jaw
        alligator_bear_open = lips < teeth < jaw
        alligator_open = alligator_bull_open or alligator_bear_open

        # Sleeping: lines tangled or within small range
        rng = max(jaw, teeth, lips) - min(jaw, teeth, lips)
        avg = (jaw + teeth + lips) / 3
        alligator_sleeping = rng / avg < 0.005 if avg > 0 else True

        # Bullish/bearish
        alligator_bullish: bool | None = current > max(jaw, teeth, lips)
        alligator_bearish: bool | None = current < min(jaw, teeth, lips)
    else:
        alligator_open = alligator_sleeping = False
        alligator_bullish = alligator_bearish = None

    # Score
    score_parts = []

    # Price position vs Alligator
    if alligator_bullish:
        score_parts.append(60.0)
    elif alligator_bearish:
        score_parts.append(-60.0)
    else:
        score_parts.append(0.0)

    # Support/resistance distance
    if res_pct is not None and sup_pct is not None:
        # Closer to support = more bullish potential
        sr_score = (res_pct - sup_pct) / (res_pct + sup_pct + 1e-9) * -100.0
        score_parts.append(sr_score * 0.4)
    elif nearest_sup is not None:
        score_parts.append(20.0)
    elif nearest_res is not None:
        score_parts.append(-20.0)

    score = sum(score_parts) / len(score_parts) * (len(score_parts) > 0)
    score = max(-100.0, min(100.0, score))

    if score >= 50:
        signal = "strong_bull"
    elif score >= 15:
        signal = "bull"
    elif score <= -50:
        signal = "strong_bear"
    elif score <= -15:
        signal = "bear"
    else:
        signal = "neutral"

    verdict = f"Williams Fractals ({symbol}): {signal.replace('_', ' ')}."
    if alligator_bullish:
        verdict += " Price above Alligator (bullish)."
    elif alligator_bearish:
        verdict += " Price below Alligator (bearish)."
    elif alligator_sleeping:
        verdict += " Alligator sleeping (consolidation)."
    if nearest_res:
        verdict += f" Nearest resistance: {nearest_res:.2f} ({res_pct:.1f}% above)."
    if nearest_sup:
        verdict += f" Nearest support: {nearest_sup:.2f} ({sup_pct:.1f}% below)."

    return WilliamsFractalReport(
        symbol=symbol,
        up_fractals=up_fractals,
        down_fractals=down_fractals,
        nearest_resistance=round(nearest_res, 4) if nearest_res else None,
        nearest_support=round(nearest_sup, 4) if nearest_sup else None,
        resistance_pct=round(res_pct, 3) if res_pct is not None else None,
        support_pct=round(sup_pct, 3) if sup_pct is not None else None,
        jaw=round(jaw, 4) if jaw else None,
        teeth=round(teeth, 4) if teeth else None,
        lips=round(lips, 4) if lips else None,
        alligator_open=alligator_open,
        alligator_sleeping=alligator_sleeping,
        alligator_bullish=alligator_bullish,
        alligator_bearish=alligator_bearish,
        score=round(score, 2),
        signal=signal,
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
