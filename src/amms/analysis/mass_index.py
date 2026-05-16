"""Mass Index Indicator.

Developed by Donald Dorsey. Identifies potential trend reversals by measuring
the narrowing and widening of the high-low range. When the range expands
significantly and then contracts, a "reversal bulge" occurs.

Calculation:
  Single EMA = EMA(High - Low, 9)
  Double EMA = EMA(Single EMA, 9)
  Mass Index Ratio = Single EMA / Double EMA
  Mass Index (MI) = SUM(ratio, 25)

Reversal bulge: MI rises above 27 then falls below 26.5
  - If trend was up before the bulge → potential top (sell signal)
  - If trend was down before the bulge → potential bottom (buy signal)

Also computes:
  - Current MI value and percentile rank
  - Bulge status (above/below thresholds)
  - Trend direction for context
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MassIndexReport:
    symbol: str

    mass_index: float       # current MI value (typical range 20-30)
    mi_pct_rank: float      # 0-100 percentile vs lookback

    # Bulge detection
    above_bulge_setup: bool    # MI currently > 27 (entering bulge zone)
    below_reversal: bool       # MI currently < 26.5 (bulge complete)
    bulge_recently: bool       # MI crossed 27 and returned below 26.5 in last N bars

    # Trend context
    trend_direction: str       # "up", "down", "flat" (from EMA slope)
    reversal_signal: str       # "none", "potential_top", "potential_bottom"

    # Normalised score and overall signal
    score: float               # -100 to +100 (high = high vol / expanding range)
    signal: str                # "expansion", "normal", "contraction"

    # History
    mi_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _ema(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    k = 2.0 / (period + 1)
    val = sum(series[:period]) / period
    result = [val]
    for x in series[period:]:
        val = x * k + val * (1 - k)
        result.append(val)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    ema_period: int = 9,
    sum_period: int = 25,
    bulge_high: float = 27.0,
    bulge_low: float = 26.5,
    history: int = 40,
    lookback: int = 60,
) -> MassIndexReport | None:
    """Compute the Mass Index.

    bars: bar objects with .high, .low, .close attributes.
    """
    min_bars = ema_period * 2 + sum_period + lookback + 5
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

    # Range series
    ranges = [highs[i] - lows[i] for i in range(n)]

    # Single EMA of range
    single_ema = _ema(ranges, ema_period)
    if not single_ema:
        return None

    # Double EMA
    double_ema = _ema(single_ema, ema_period)
    if not double_ema:
        return None

    # Ratio series (aligned from end)
    min_len = min(len(single_ema), len(double_ema))
    ratio_series = []
    for i in range(min_len):
        se = single_ema[-(min_len - i)]
        de = double_ema[-(min_len - i)]
        ratio_series.append(se / de if de > 1e-9 else 1.0)

    # Mass Index = rolling sum of ratio over sum_period
    if len(ratio_series) < sum_period:
        return None

    mi_series_full = [
        sum(ratio_series[i:i + sum_period])
        for i in range(len(ratio_series) - sum_period + 1)
    ]

    if not mi_series_full:
        return None

    cur_mi = mi_series_full[-1]

    # Percentile rank vs lookback
    mi_window = mi_series_full[-lookback:] if len(mi_series_full) >= lookback else mi_series_full
    pct_rank = sum(1 for v in mi_window if v <= cur_mi) / len(mi_window) * 100.0

    # Bulge analysis
    above_bulge = cur_mi > bulge_high
    below_rev   = cur_mi < bulge_low

    # Did MI cross above bulge_high and then back below bulge_low recently?
    bulge_recently = False
    recent_mi = mi_series_full[-(history + 1):]  # last ~40 bars
    was_above = False
    for v in recent_mi[:-1]:
        if v > bulge_high:
            was_above = True
        if was_above and v < bulge_low:
            bulge_recently = True
            break

    # Trend direction from EMA slope
    if len(closes) >= 20:
        ema20 = _ema(closes, 20)
        if len(ema20) >= 2:
            slope = ema20[-1] - ema20[-2]
            trend_dir = "up" if slope > 0 else ("down" if slope < 0 else "flat")
        else:
            trend_dir = "flat"
    else:
        trend_dir = "flat"

    # Reversal signal
    if bulge_recently:
        if trend_dir == "up":
            reversal_signal = "potential_top"
        elif trend_dir == "down":
            reversal_signal = "potential_bottom"
        else:
            reversal_signal = "none"
    else:
        reversal_signal = "none"

    # Score: high MI = range expansion, low = contraction
    # Map pct_rank to -100..+100 (expansion positive, contraction negative)
    score = (pct_rank - 50) * 2.0
    score = max(-100.0, min(100.0, score))

    # Simple signal
    if cur_mi > bulge_high:
        signal = "expansion"
    elif cur_mi < bulge_low:
        signal = "contraction"
    else:
        signal = "normal"

    hist_series = mi_series_full[-history:]

    verdict = (
        f"Mass Index ({symbol}): MI={cur_mi:.2f}, rank={pct_rank:.0f}th pct. "
        f"Range {signal}. Trend: {trend_dir}."
    )
    if reversal_signal != "none":
        verdict += f" Reversal bulge: {reversal_signal.replace('_', ' ')}."

    return MassIndexReport(
        symbol=symbol,
        mass_index=round(cur_mi, 3),
        mi_pct_rank=round(pct_rank, 1),
        above_bulge_setup=above_bulge,
        below_reversal=below_rev,
        bulge_recently=bulge_recently,
        trend_direction=trend_dir,
        reversal_signal=reversal_signal,
        score=round(score, 2),
        signal=signal,
        mi_series=[round(v, 3) for v in hist_series],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
