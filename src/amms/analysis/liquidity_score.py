"""Liquidity scoring for symbols.

Evaluates how liquid a symbol is based on observable bar data:
  1. Volume level (0-30 pts): average daily volume vs minimum threshold
  2. Volume consistency (0-25 pts): low coefficient of variation = consistent
  3. Spread proxy (0-25 pts): average (high-low)/close — tighter = more liquid
  4. Volume trend (0-20 pts): is volume increasing or holding vs declining?

Total: 0-100 (higher = more liquid)
  80-100: highly liquid
  60-79: liquid
  40-59: moderate
  20-39: low liquidity
  0-19: illiquid / avoid

This is a relative, data-driven measure and does NOT compare against
exchange-level bid/ask spread (not available from bar data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityScore:
    symbol: str
    total_score: float          # 0-100
    grade: str                  # "A"|"B"|"C"|"D"|"F"
    avg_volume: float
    volume_cv: float            # coefficient of variation (lower = more consistent)
    avg_spread_pct: float       # avg (high-low)/close * 100 (lower = tighter)
    volume_trend: str           # "rising"|"stable"|"declining"
    vol_score: float            # 0-30
    consistency_score: float    # 0-25
    spread_score: float         # 0-25
    trend_score: float          # 0-20
    bars_used: int


def score(bars: list, *, lookback: int = 30) -> LiquidityScore | None:
    """Compute liquidity score for a symbol.

    bars: list[Bar] — needs at least 10 bars
    lookback: number of recent bars to analyze
    Returns None if insufficient data.
    """
    if len(bars) < 10:
        return None

    symbol = bars[0].symbol
    window = bars[-lookback:] if len(bars) > lookback else bars
    n = len(window)

    volumes = [b.volume for b in window]
    avg_vol = sum(volumes) / n

    # 1. Volume level score (0-30): relative to 100k shares/day baseline
    vol_baseline = 100_000
    vol_ratio = avg_vol / vol_baseline if vol_baseline > 0 else 0
    if vol_ratio >= 10:
        vol_score = 30.0
    elif vol_ratio >= 3:
        vol_score = 25.0
    elif vol_ratio >= 1:
        vol_score = 20.0
    elif vol_ratio >= 0.3:
        vol_score = 12.0
    elif vol_ratio >= 0.1:
        vol_score = 5.0
    else:
        vol_score = 0.0

    # 2. Volume consistency (0-25): coefficient of variation
    if avg_vol > 0:
        vol_std = math.sqrt(sum((v - avg_vol) ** 2 for v in volumes) / n)
        vol_cv = vol_std / avg_vol
    else:
        vol_cv = float("inf")

    if vol_cv <= 0.2:
        consistency_score = 25.0
    elif vol_cv <= 0.4:
        consistency_score = 20.0
    elif vol_cv <= 0.7:
        consistency_score = 14.0
    elif vol_cv <= 1.2:
        consistency_score = 8.0
    else:
        consistency_score = 2.0

    # 3. Spread proxy (0-25): avg (high-low)/close
    spread_pcts = []
    for b in window:
        if b.close > 0:
            spread_pcts.append((b.high - b.low) / b.close * 100)
    avg_spread = sum(spread_pcts) / len(spread_pcts) if spread_pcts else 0.0

    if avg_spread <= 0.5:
        spread_score = 25.0
    elif avg_spread <= 1.0:
        spread_score = 20.0
    elif avg_spread <= 2.0:
        spread_score = 14.0
    elif avg_spread <= 4.0:
        spread_score = 7.0
    else:
        spread_score = 2.0

    # 4. Volume trend (0-20): compare first half to second half
    half = max(1, n // 2)
    first_half_avg = sum(volumes[:half]) / half
    second_half_avg = sum(volumes[half:]) / max(1, n - half)

    if first_half_avg > 0:
        trend_ratio = second_half_avg / first_half_avg
    else:
        trend_ratio = 1.0

    if trend_ratio >= 1.1:
        volume_trend = "rising"
        trend_score = 20.0
    elif trend_ratio >= 0.85:
        volume_trend = "stable"
        trend_score = 14.0
    else:
        volume_trend = "declining"
        trend_score = 5.0

    total = vol_score + consistency_score + spread_score + trend_score

    if total >= 80:
        grade = "A"
    elif total >= 60:
        grade = "B"
    elif total >= 40:
        grade = "C"
    elif total >= 20:
        grade = "D"
    else:
        grade = "F"

    return LiquidityScore(
        symbol=symbol,
        total_score=round(total, 1),
        grade=grade,
        avg_volume=round(avg_vol, 0),
        volume_cv=round(vol_cv, 3),
        avg_spread_pct=round(avg_spread, 3),
        volume_trend=volume_trend,
        vol_score=round(vol_score, 1),
        consistency_score=round(consistency_score, 1),
        spread_score=round(spread_score, 1),
        trend_score=round(trend_score, 1),
        bars_used=n,
    )
