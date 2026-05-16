"""Relative Vigor Index (RVI).

Developed by John Ehlers. Based on the observation that in a rising market,
prices tend to close higher than they open (vigor), and in falling markets,
prices tend to close lower.

Numerator:  CLOSE - OPEN (the bar's directional vigor)
Denominator: HIGH - LOW  (the bar's total range)

Smoothing uses a 4-bar symmetric weighted average (SWA):
  SWA(x) = (x + 2×x[1] + 2×x[2] + x[3]) / 6

RVI = SMA(SWA(numerator), period) / SMA(SWA(denominator), period)
Signal line = 4-bar SWA of RVI (same symmetric smoothing)

Histogram = RVI - Signal

Range: roughly -1 to +1
  RVI > 0  → bullish vigor
  RVI < 0  → bearish vigor
  RVI > Signal → bullish momentum
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RVIReport:
    symbol: str
    period: int

    rvi: float              # current RVI value
    rvi_signal: float       # signal line (4-bar SWA of RVI)
    rvi_histogram: float

    bullish: bool           # RVI > 0
    above_signal: bool      # RVI > signal

    cross_up: bool
    cross_down: bool

    # Score
    score: float            # -100 to +100
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # History
    rvi_series: list[float]
    signal_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _swa4(series: list[float]) -> list[float]:
    """4-bar symmetric weighted average: (x + 2x[1] + 2x[2] + x[3]) / 6."""
    if len(series) < 4:
        return []
    result = []
    for i in range(3, len(series)):
        val = (series[i] + 2 * series[i - 1] + 2 * series[i - 2] + series[i - 3]) / 6.0
        result.append(val)
    return result


def _sma(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    return [sum(series[i:i + period]) / period for i in range(len(series) - period + 1)]


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 10,
    history: int = 15,
) -> RVIReport | None:
    """Compute the Relative Vigor Index.

    bars: bar objects with .open, .high, .low, .close attributes.
    """
    min_bars = period + 4 + 4 + history + 5  # SWA needs 4 extra, signal SWA needs 4 more
    if not bars or len(bars) < min_bars:
        return None

    try:
        opens  = [float(b.open)  for b in bars]
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Numerator: close - open; Denominator: high - low
    numerator   = [closes[i] - opens[i] for i in range(n)]
    denominator = [highs[i]  - lows[i]  for i in range(n)]

    # Apply SWA smoothing
    swa_num = _swa4(numerator)
    swa_den = _swa4(denominator)

    if not swa_num or not swa_den:
        return None

    # SMA of smoothed numerator and denominator
    sma_num = _sma(swa_num, period)
    sma_den = _sma(swa_den, period)

    # RVI = sma_num / sma_den (aligned from end)
    min_len = min(len(sma_num), len(sma_den))
    rvi_vals = []
    for i in range(min_len):
        d = sma_den[-(min_len - i)]
        r = sma_num[-(min_len - i)]
        rvi_vals.append(r / d if abs(d) > 1e-9 else 0.0)

    if len(rvi_vals) < 5:
        return None

    # Signal = SWA4 of RVI
    sig_vals = _swa4(rvi_vals)
    if not sig_vals:
        return None

    cur_rvi = rvi_vals[-1]
    cur_sig = sig_vals[-1]
    rvi_hist = cur_rvi - cur_sig

    # Cross detection
    if len(rvi_vals) >= 2 and len(sig_vals) >= 2:
        prev_rvi = rvi_vals[-2]
        prev_sig = sig_vals[-2]
        cross_up   = cur_rvi >= cur_sig and prev_rvi < prev_sig
        cross_down = cur_rvi <= cur_sig and prev_rvi > prev_sig
    else:
        cross_up = cross_down = False

    # Score: normalise RVI and histogram
    rvi_window = rvi_vals[-30:] if len(rvi_vals) >= 30 else rvi_vals
    rvi_max = max(abs(v) for v in rvi_window) if rvi_window else 1.0
    rvi_norm = max(-100.0, min(100.0, cur_rvi / rvi_max * 100.0)) if rvi_max > 1e-9 else 0.0

    hist_max = max(abs(rvi_vals[-i] - sig_vals[-i]) for i in range(1, min(len(sig_vals), 10) + 1)) if sig_vals else 1.0
    hist_norm = max(-100.0, min(100.0, rvi_hist / hist_max * 100.0)) if hist_max > 1e-9 else 0.0

    score = rvi_norm * 0.6 + hist_norm * 0.4
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

    hist_rvi = rvi_vals[-history:]
    hist_sig = sig_vals[-history:]

    verdict = (
        f"RVI ({symbol}, {period}): {signal.replace('_', ' ')}. "
        f"RVI={cur_rvi:+.4f}, Signal={cur_sig:+.4f}, Hist={rvi_hist:+.4f}."
    )
    if cross_up:
        verdict += " Bullish cross."
    if cross_down:
        verdict += " Bearish cross."

    return RVIReport(
        symbol=symbol,
        period=period,
        rvi=round(cur_rvi, 5),
        rvi_signal=round(cur_sig, 5),
        rvi_histogram=round(rvi_hist, 5),
        bullish=cur_rvi > 0,
        above_signal=cur_rvi > cur_sig,
        cross_up=cross_up,
        cross_down=cross_down,
        score=round(score, 2),
        signal=signal,
        rvi_series=[round(v, 5) for v in hist_rvi],
        signal_series=[round(v, 5) for v in hist_sig],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
