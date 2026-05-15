"""Stochastic Oscillator (%K and %D).

%K = (close - lowest_low_n) / (highest_high_n - lowest_low_n) * 100
%D = SMA(m) of %K  (signal line, typically m=3)

Interpretation:
  %K < 20 : oversold
  %K > 80 : overbought
  %K crosses %D upward below 20 : bullish signal
  %K crosses %D downward above 80 : bearish signal
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class StochasticResult:
    k: float    # %K (raw stochastic 0..100)
    d: float    # %D (smoothed signal, SMA of %K)
    zone: str   # "oversold" | "overbought" | "neutral"
    signal: str  # "bullish_cross" | "bearish_cross" | "none"


def stochastic(bars: list[Bar], k_period: int = 14, d_period: int = 3) -> StochasticResult | None:
    """Compute Stochastic %K and %D.

    k_period : lookback for highest high / lowest low (default 14)
    d_period : smoothing for %D signal line (default 3)
    Returns None if insufficient data.
    """
    if k_period < 1:
        raise ValueError(f"k_period must be >= 1, got {k_period}")
    if d_period < 1:
        raise ValueError(f"d_period must be >= 1, got {d_period}")
    if len(bars) < k_period + d_period:
        return None

    # Compute %K for last (d_period + 1) bars so we can detect crossovers
    k_values: list[float] = []
    needed = k_period + d_period
    recent = bars[-(needed):]
    for i in range(k_period - 1, len(recent)):
        window = recent[i - k_period + 1: i + 1]
        low_n = min(b.low for b in window)
        high_n = max(b.high for b in window)
        close = recent[i].close
        spread = high_n - low_n
        k = (close - low_n) / spread * 100.0 if spread > 0 else 50.0
        k_values.append(k)

    if len(k_values) < d_period:
        return None

    # %D = SMA of last d_period %K values
    d = sum(k_values[-d_period:]) / d_period
    k = k_values[-1]
    k_prev = k_values[-2] if len(k_values) >= 2 else k
    d_prev = sum(k_values[-(d_period + 1):-1]) / d_period if len(k_values) >= d_period + 1 else d

    if k < 20:
        zone = "oversold"
    elif k > 80:
        zone = "overbought"
    else:
        zone = "neutral"

    # Crossover detection
    if k_prev <= d_prev and k > d and zone == "oversold":
        signal = "bullish_cross"
    elif k_prev >= d_prev and k < d and zone == "overbought":
        signal = "bearish_cross"
    else:
        signal = "none"

    return StochasticResult(
        k=round(k, 2),
        d=round(d, 2),
        zone=zone,
        signal=signal,
    )
