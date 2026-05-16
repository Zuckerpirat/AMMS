"""Vortex Indicator.

Developed by Etienne Botes and Douglas Siepman (2010). Identifies the start
of new trends and confirms ongoing trends using two oscillators:

  VM+ (Positive Vortex Movement) = |High_t - Low_{t-1}|
  VM- (Negative Vortex Movement) = |Low_t  - High_{t-1}|

Normalised over N periods by True Range:
  VI+ = sum(VM+, n) / sum(TR, n)
  VI- = sum(VM-, n) / sum(TR, n)

Signals:
  VI+ > VI-  → bullish trend
  VI+ > 1.1  → strong bull confirmation
  VI- > 1.1  → strong bear confirmation
  Crossover  → potential trend reversal

Also computes a "vortex spread" (VI+ - VI-) as a continuous momentum score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VortexReport:
    symbol: str
    period: int

    # Core components
    vi_plus: float      # 0-1.5+; above 1.0 = strong positive movement
    vi_minus: float     # 0-1.5+; above 1.0 = strong negative movement
    vortex_spread: float  # VI+ - VI-  (-1.5 to +1.5)

    # Trend
    bullish: bool       # VI+ > VI-
    vi_plus_above_threshold: bool   # VI+ > 1.1
    vi_minus_above_threshold: bool  # VI- > 1.1

    # Crossover detection (last few bars)
    cross_bullish: bool    # VI+ just crossed above VI-
    cross_bearish: bool    # VI- just crossed above VI+

    # Normalised score and signal
    score: float        # -100 to +100
    signal: str         # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # Series (last 10 readings)
    vi_plus_series: list[float]
    vi_minus_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _true_range(highs: list[float], lows: list[float], closes: list[float], i: int) -> float:
    if i == 0:
        return highs[i] - lows[i]
    return max(
        highs[i] - lows[i],
        abs(highs[i] - closes[i - 1]),
        abs(lows[i] - closes[i - 1]),
    )


def _vortex_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> tuple[list[float], list[float]]:
    """Return (VI+_series, VI-_series), starting from index `period`."""
    n = len(closes)
    if n < period + 1:
        return [], []

    vi_plus_list = []
    vi_minus_list = []

    for end in range(period, n):
        # Sum VM+ and VM- over [end-period+1 .. end]
        sum_vmp = 0.0
        sum_vmm = 0.0
        sum_tr  = 0.0
        for i in range(end - period + 1, end + 1):
            sum_vmp += abs(highs[i]   - lows[i - 1])
            sum_vmm += abs(lows[i]    - highs[i - 1])
            sum_tr  += _true_range(highs, lows, closes, i)

        if sum_tr < 1e-9:
            vi_plus_list.append(1.0)
            vi_minus_list.append(1.0)
        else:
            vi_plus_list.append(sum_vmp / sum_tr)
            vi_minus_list.append(sum_vmm / sum_tr)

    return vi_plus_list, vi_minus_list


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 14,
    threshold: float = 1.1,
    history: int = 10,
) -> VortexReport | None:
    """Compute the Vortex Indicator.

    bars: bar objects with .high, .low, .close attributes.
    period: lookback for VI+/VI- calculation.
    """
    min_bars = period + history + 3
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

    vi_plus_all, vi_minus_all = _vortex_series(highs, lows, closes, period)
    if len(vi_plus_all) < 2:
        return None

    cur_vip = vi_plus_all[-1]
    cur_vim = vi_minus_all[-1]
    spread  = cur_vip - cur_vim

    bullish = cur_vip > cur_vim
    vip_thresh = cur_vip > threshold
    vim_thresh = cur_vim > threshold

    # Crossover detection (last 2 readings)
    prev_vip = vi_plus_all[-2]
    prev_vim = vi_minus_all[-2]
    cross_bull = (cur_vip >= cur_vim) and (prev_vip < prev_vim)
    cross_bear = (cur_vim >= cur_vip) and (prev_vim < prev_vip)

    # Score: map spread to -100..+100
    # Typical spread range is about -0.5 to +0.5; scale by 200
    score = max(-100.0, min(100.0, spread * 200.0))

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

    # History
    hist_vip = vi_plus_all[-history:]
    hist_vim = vi_minus_all[-history:]

    verdict = (
        f"Vortex ({symbol}, {period}): {signal.replace('_', ' ')}. "
        f"VI+={cur_vip:.3f}, VI-={cur_vim:.3f}, spread={spread:+.3f}."
    )
    if cross_bull:
        verdict += " Bullish cross."
    if cross_bear:
        verdict += " Bearish cross."

    return VortexReport(
        symbol=symbol,
        period=period,
        vi_plus=round(cur_vip, 4),
        vi_minus=round(cur_vim, 4),
        vortex_spread=round(spread, 4),
        bullish=bullish,
        vi_plus_above_threshold=vip_thresh,
        vi_minus_above_threshold=vim_thresh,
        cross_bullish=cross_bull,
        cross_bearish=cross_bear,
        score=round(score, 2),
        signal=signal,
        vi_plus_series=[round(v, 4) for v in hist_vip],
        vi_minus_series=[round(v, 4) for v in hist_vim],
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
