"""Williams %R oscillator.

Williams %R = (highest_high_n - close) / (highest_high_n - lowest_low_n) * -100

Range: 0 to -100
  -80 to -100 → oversold (potential buy)
  -20 to   0  → overbought (potential sell)
  -50         → mid-point (neutral)

Unlike Stochastic %K, Williams %R uses the inverse formula (negative scale)
and is not smoothed by default — it reacts faster.

Optional smoothing: an EMA of %R values to reduce noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class WilliamsRResult:
    value: float          # current %R, range -100..0
    smoothed: float       # EMA-smoothed %R (same range)
    zone: str             # "overbought" | "oversold" | "neutral"
    signal: str           # "buy" | "sell" | "none"
    period: int


def williams_r(bars: list[Bar], period: int = 14, smooth: int = 3) -> WilliamsRResult | None:
    """Compute Williams %R for the last bar.

    period: lookback for highest-high / lowest-low
    smooth: EMA period for the smoothed line (1 = no smoothing)
    Returns None if insufficient data.
    """
    need = period + max(smooth - 1, 0)
    if len(bars) < need:
        return None

    # Compute raw %R series over the last (smooth + period - 1) bars
    r_series: list[float] = []
    start = len(bars) - (smooth + period - 1)
    if start < 0:
        start = 0

    for i in range(start + period - 1, len(bars)):
        window = bars[i - period + 1: i + 1]
        hh = max(b.high for b in window)
        ll = min(b.low for b in window)
        close = bars[i].close
        if hh == ll:
            r_val = -50.0
        else:
            r_val = (hh - close) / (hh - ll) * -100.0
        r_series.append(r_val)

    if not r_series:
        return None

    current_r = r_series[-1]

    # EMA smoothing
    if smooth > 1 and len(r_series) >= smooth:
        k = 2.0 / (smooth + 1)
        ema = sum(r_series[:smooth]) / smooth
        for v in r_series[smooth:]:
            ema = v * k + ema * (1 - k)
        smoothed = ema
    else:
        smoothed = current_r

    if current_r >= -20.0:
        zone = "overbought"
        signal = "sell"
    elif current_r <= -80.0:
        zone = "oversold"
        signal = "buy"
    else:
        zone = "neutral"
        signal = "none"

    return WilliamsRResult(
        value=round(current_r, 2),
        smoothed=round(smoothed, 2),
        zone=zone,
        signal=signal,
        period=period,
    )
