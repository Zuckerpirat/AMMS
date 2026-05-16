"""Chande Momentum Oscillator (CMO).

Developed by Tushar Chande. Similar to RSI but uses the raw sum of up and
down moves rather than averages:

  CMO = (SumUp - SumDown) / (SumUp + SumDown) × 100

Where:
  SumUp   = sum of positive price changes over N periods
  SumDown = sum of absolute negative price changes over N periods

Range: -100 (extreme oversold) to +100 (extreme overbought)

Signals:
  CMO > +50  → overbought (potential reversal)
  CMO < -50  → oversold   (potential reversal)
  CMO > 0    → bullish momentum
  CMO crossing 0 → momentum shift

Also computes:
  - Rolling CMO series for trend analysis
  - CMO smoothed with 9-bar EMA (signal line)
  - Absolute CMO (volatility proxy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CMOReport:
    symbol: str
    period: int

    cmo: float          # current CMO (-100 to +100)
    cmo_signal: float   # EMA-9 of CMO (signal line)
    cmo_histogram: float  # CMO - signal

    # State
    bullish: bool       # CMO > 0
    overbought: bool    # CMO > +50
    oversold: bool      # CMO < -50
    above_signal: bool  # CMO > signal line

    # Cross detection
    zero_cross_up: bool
    zero_cross_down: bool
    signal_cross_up: bool
    signal_cross_down: bool

    # Volatility
    abs_cmo: float      # |CMO| — measures trend strength/choppiness

    # Score and signal label
    score: float        # -100 to +100 (same as CMO but adjusted for signal)
    signal: str         # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # History
    cmo_series: list[float]
    signal_series: list[float]

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


def _cmo_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period + 1:
        return []
    result = []
    for i in range(period, len(closes)):
        window = closes[i - period:i + 1]
        sum_up = 0.0
        sum_down = 0.0
        for j in range(1, len(window)):
            diff = window[j] - window[j - 1]
            if diff > 0:
                sum_up += diff
            else:
                sum_down += abs(diff)
        total = sum_up + sum_down
        if total > 1e-9:
            result.append((sum_up - sum_down) / total * 100.0)
        else:
            result.append(0.0)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 20,
    signal_period: int = 9,
    history: int = 20,
) -> CMOReport | None:
    """Compute the Chande Momentum Oscillator.

    bars: bar objects with .close attribute.
    """
    min_bars = period + signal_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # CMO series
    cmo_vals = _cmo_series(closes, period)
    if len(cmo_vals) < signal_period + 2:
        return None

    # Signal line
    sig_vals = _ema(cmo_vals, signal_period)
    if not sig_vals:
        return None

    cur_cmo = cmo_vals[-1]
    cur_sig = sig_vals[-1]
    cmo_hist = cur_cmo - cur_sig

    bullish   = cur_cmo > 0
    overbought = cur_cmo > 50
    oversold   = cur_cmo < -50
    above_sig  = cur_cmo > cur_sig

    # Cross detection (last 2 bars)
    if len(cmo_vals) >= 2:
        prev_cmo = cmo_vals[-2]
        zero_cross_up   = cur_cmo >= 0 and prev_cmo < 0
        zero_cross_down = cur_cmo <= 0 and prev_cmo > 0
    else:
        zero_cross_up = zero_cross_down = False

    if len(cmo_vals) >= 2 and len(sig_vals) >= 2:
        prev_cmo2 = cmo_vals[-2]
        prev_sig  = sig_vals[-2]
        sig_cross_up   = cur_cmo >= cur_sig and prev_cmo2 < prev_sig
        sig_cross_down = cur_cmo <= cur_sig and prev_cmo2 > prev_sig
    else:
        sig_cross_up = sig_cross_down = False

    # Score (CMO is already -100..+100)
    score = max(-100.0, min(100.0, cur_cmo * 0.7 + cmo_hist * 0.3))

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
    hist_cmo = cmo_vals[-history:]
    hist_sig = sig_vals[-history:]

    verdict = (
        f"CMO ({symbol}, {period}): {signal.replace('_', ' ')}. "
        f"CMO={cur_cmo:+.1f}, Signal={cur_sig:+.1f}."
    )
    if overbought:
        verdict += " Overbought (>+50)."
    if oversold:
        verdict += " Oversold (<-50)."
    if zero_cross_up:
        verdict += " Zero-line bullish cross."
    if zero_cross_down:
        verdict += " Zero-line bearish cross."

    return CMOReport(
        symbol=symbol,
        period=period,
        cmo=round(cur_cmo, 2),
        cmo_signal=round(cur_sig, 2),
        cmo_histogram=round(cmo_hist, 2),
        bullish=bullish,
        overbought=overbought,
        oversold=oversold,
        above_signal=above_sig,
        zero_cross_up=zero_cross_up,
        zero_cross_down=zero_cross_down,
        signal_cross_up=sig_cross_up,
        signal_cross_down=sig_cross_down,
        abs_cmo=round(abs(cur_cmo), 2),
        score=round(score, 2),
        signal=signal,
        cmo_series=[round(v, 2) for v in hist_cmo],
        signal_series=[round(v, 2) for v in hist_sig],
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
