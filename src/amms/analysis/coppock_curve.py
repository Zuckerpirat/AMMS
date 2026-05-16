"""Coppock Curve.

Developed by Edwin Coppock (1962). Originally designed for monthly data as a
long-term buy signal for stock market indices. Adapted here for daily bars.

Calculation (daily adaptation):
  ROC1 = Rate of Change over roc1_period bars (default 294 = ~14 months daily)
  ROC2 = Rate of Change over roc2_period bars (default 231 = ~11 months daily)
  Coppock = WMA(ROC1 + ROC2, wma_period)  (default 10-month WMA = 210 days)

For shorter-term use, also provides scaled-down parameters.

Interpretation:
  Buy signal:  Coppock turns UP while below zero
  Hold signal: Coppock continues rising (above zero = still valid)
  No short:    Coppock was originally buy-only; bearish when falling from above zero

Also tracks:
  - Direction (rising/falling)
  - Zero-line position
  - Rate of change of the curve (acceleration/deceleration)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CoppockReport:
    symbol: str

    coppock: float          # current value
    coppock_prev: float     # previous bar value
    rising: bool            # coppock > coppock_prev
    above_zero: bool        # coppock > 0

    # Key signals
    buy_signal: bool        # turning up while below zero
    sell_alert: bool        # turning down while above zero

    # Acceleration
    acceleration: float     # rate of change of the curve (second derivative)

    # Normalised score
    score: float            # -100 to +100
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # History (last 20 values)
    coppock_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _roc(series: list[float], period: int) -> list[float]:
    result = []
    for i in range(period, len(series)):
        prev = series[i - period]
        if prev > 1e-9:
            result.append((series[i] / prev - 1.0) * 100.0)
        else:
            result.append(0.0)
    return result


def _wma(series: list[float], period: int) -> list[float]:
    """Weighted Moving Average."""
    if len(series) < period:
        return []
    weights = list(range(1, period + 1))
    total_w = sum(weights)
    result = []
    for i in range(period - 1, len(series)):
        wsum = sum(weights[j] * series[i - (period - 1) + j] for j in range(period))
        result.append(wsum / total_w)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    roc1_period: int = 294,
    roc2_period: int = 231,
    wma_period: int = 210,
    short_mode: bool = False,
    history: int = 20,
) -> CoppockReport | None:
    """Compute the Coppock Curve.

    bars: bar objects with .close attribute.

    short_mode: if True, uses shorter periods (roc1=14, roc2=11, wma=10)
                suitable for weekly/tactical analysis.
    """
    if short_mode:
        roc1_period = 14
        roc2_period = 11
        wma_period  = 10

    min_bars = max(roc1_period, roc2_period) + wma_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # ROC1 and ROC2
    roc1 = _roc(closes, roc1_period)
    roc2_aligned = _roc(closes, roc2_period)

    # Align lengths (take from end)
    min_len = min(len(roc1), len(roc2_aligned))
    if min_len < wma_period + 2:
        return None

    combined = [roc1[-(min_len - i)] + roc2_aligned[-(min_len - i)] for i in range(min_len)]

    # WMA of combined
    coppock_vals = _wma(combined, wma_period)
    if len(coppock_vals) < 3:
        return None

    cur = coppock_vals[-1]
    prev = coppock_vals[-2]
    prev2 = coppock_vals[-3]

    rising = cur > prev
    above_zero = cur > 0

    # Key signals
    buy_signal = rising and not above_zero and not (prev > 0)
    sell_alert = not rising and above_zero

    # Acceleration = first diff of second diff
    accel = (cur - prev) - (prev - prev2)

    # Normalised score
    cop_window = coppock_vals[-50:] if len(coppock_vals) >= 50 else coppock_vals
    cop_max = max(abs(v) for v in cop_window) if cop_window else 1.0
    cop_norm = max(-100.0, min(100.0, cur / cop_max * 100.0)) if cop_max > 1e-9 else 0.0

    # Direction bonus
    dir_bonus = 20.0 if rising else -20.0
    score = cop_norm * 0.8 + dir_bonus * 0.2
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

    hist_series = coppock_vals[-history:]

    direction_str = "rising" if rising else "falling"
    zero_str = "above zero" if above_zero else "below zero"

    verdict = (
        f"Coppock Curve ({symbol}): {signal.replace('_', ' ')}. "
        f"Value={cur:+.3f} ({direction_str}, {zero_str})."
    )
    if buy_signal:
        verdict += " BUY SIGNAL: turning up from below zero."
    if sell_alert:
        verdict += " SELL ALERT: turning down from above zero."

    return CoppockReport(
        symbol=symbol,
        coppock=round(cur, 4),
        coppock_prev=round(prev, 4),
        rising=rising,
        above_zero=above_zero,
        buy_signal=buy_signal,
        sell_alert=sell_alert,
        acceleration=round(accel, 4),
        score=round(score, 2),
        signal=signal,
        coppock_series=[round(v, 4) for v in hist_series],
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
