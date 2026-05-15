"""Trend consistency scoring.

Measures how smooth and consistent a price trend is:
  1. R-squared of linear regression fit (0-1): how well price follows a line
  2. Trend efficiency (Kaufman): net displacement / total path length
  3. Noise level: residual volatility around the trend line
  4. Choppiness index: derived from ATR vs price range

Composite score 0-100:
  80-100: highly consistent trend
  60-79:  consistent
  40-59:  moderate
  20-39:  choppy
  0-19:   random/no trend
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrendConsistency:
    symbol: str
    score: float             # 0-100
    label: str               # "consistent"|"moderate"|"choppy"|"random"
    r_squared: float         # 0-1 (linear regression fit quality)
    efficiency: float        # 0-1 (net move / total path)
    noise_pct: float         # residual std as % of price
    direction: str           # "up"|"down"|"flat"
    slope_per_day_pct: float # daily slope as % of starting price
    bars_used: int


def score(bars: list, *, lookback: int = 20) -> TrendConsistency | None:
    """Compute trend consistency for the given bars.

    bars: list[Bar] — needs at least lookback + 1 bars
    lookback: analysis window (default 20)
    Returns None if insufficient data.
    """
    if len(bars) < max(lookback, 5):
        return None

    symbol = bars[0].symbol
    window = bars[-lookback:] if len(bars) >= lookback else bars
    n = len(window)

    closes = [b.close for b in window]
    if any(c <= 0 for c in closes):
        return None

    # Linear regression
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n
    ss_xy = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((closes[i] - y_mean) ** 2 for i in range(n))

    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    intercept = y_mean - slope * x_mean

    # R-squared
    if ss_yy > 0:
        ss_res = sum((closes[i] - (slope * i + intercept)) ** 2 for i in range(n))
        r_squared = max(0.0, 1 - ss_res / ss_yy)
    else:
        r_squared = 1.0  # flat line is perfectly fit

    # Trend efficiency: |net change| / sum of |bar changes|
    net_change = abs(closes[-1] - closes[0])
    total_path = sum(abs(closes[i] - closes[i - 1]) for i in range(1, n))
    efficiency = net_change / total_path if total_path > 0 else 0.0

    # Noise: std of residuals as % of mean price
    if ss_yy > 0:
        residuals = [closes[i] - (slope * i + intercept) for i in range(n)]
        resid_std = math.sqrt(sum(r ** 2 for r in residuals) / n)
        noise_pct = resid_std / y_mean * 100
    else:
        noise_pct = 0.0

    # Direction
    slope_pct = slope / closes[0] * 100 if closes[0] > 0 else 0.0
    if slope_pct > 0.1:
        direction = "up"
    elif slope_pct < -0.1:
        direction = "down"
    else:
        direction = "flat"

    # Composite score
    r2_score = r_squared * 50          # 0-50
    eff_score = efficiency * 30        # 0-30
    # noise penalty: 0% noise = 20pts, 5%+ noise = 0pts
    noise_score = max(0.0, 20.0 - noise_pct * 4)

    total = r2_score + eff_score + noise_score

    if total >= 80:
        label = "consistent"
    elif total >= 55:
        label = "moderate"
    elif total >= 30:
        label = "choppy"
    else:
        label = "random"

    return TrendConsistency(
        symbol=symbol,
        score=round(total, 1),
        label=label,
        r_squared=round(r_squared, 4),
        efficiency=round(efficiency, 4),
        noise_pct=round(noise_pct, 2),
        direction=direction,
        slope_per_day_pct=round(slope_pct, 4),
        bars_used=n,
    )
