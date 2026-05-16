"""Kaufman's Adaptive Moving Average (KAMA).

Developed by Perry Kaufman. KAMA adapts its smoothing speed based on the
market's efficiency ratio (ER):

  Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|, n)
  ER near 1 = efficient trend; ER near 0 = choppy market

  Smoothing Constant (SC) = (ER × (fast_sc - slow_sc) + slow_sc)²
    where fast_sc = 2/(fast+1), slow_sc = 2/(slow+1)
    default: fast=2, slow=30

  KAMA[t] = KAMA[t-1] + SC × (Close[t] - KAMA[t-1])

Outputs:
  - KAMA value (adaptive trend line)
  - Efficiency Ratio (0-1)
  - Smoothing Constant
  - Direction and rate of change
  - Signal: price vs KAMA position
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KAMAReport:
    symbol: str

    kama: float             # current KAMA value
    efficiency_ratio: float  # 0-1 (higher = more efficient/trending)
    smoothing_constant: float  # current SC (higher = faster adaptation)

    # Price relationship
    price_above_kama: bool
    price_distance_pct: float   # (price - kama) / kama * 100

    # KAMA direction
    kama_rising: bool
    kama_slope_pct: float    # rate of change of KAMA as % (last 5 bars)

    # Market regime from ER
    er_regime: str           # "trending", "choppy", "transitional"

    # Score and signal
    score: float             # -100 to +100
    signal: str              # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # History (last 20 bars)
    kama_series: list[float]
    er_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _compute_kama(
    closes: list[float],
    period: int,
    fast: int,
    slow: int,
) -> list[float]:
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    if len(closes) < period + 1:
        return []

    kama = closes[period - 1]  # initialise at first full period close
    kama_vals = [kama]

    for i in range(period, len(closes)):
        direction = abs(closes[i] - closes[i - period])
        volatility = sum(abs(closes[j] - closes[j - 1]) for j in range(i - period + 1, i + 1))
        er = direction / volatility if volatility > 1e-9 else 0.0
        er = max(0.0, min(1.0, er))
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + sc * (closes[i] - kama)
        kama_vals.append(kama)

    return kama_vals


def _compute_er_series(closes: list[float], period: int) -> list[float]:
    result = []
    for i in range(period, len(closes)):
        direction = abs(closes[i] - closes[i - period])
        volatility = sum(abs(closes[j] - closes[j - 1]) for j in range(i - period + 1, i + 1))
        er = direction / volatility if volatility > 1e-9 else 0.0
        result.append(max(0.0, min(1.0, er)))
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 10,
    fast: int = 2,
    slow: int = 30,
    history: int = 20,
) -> KAMAReport | None:
    """Compute Kaufman's Adaptive Moving Average.

    bars: bar objects with .close attribute.
    period: efficiency ratio lookback.
    fast/slow: fast and slow EMA periods for SC computation.
    """
    min_bars = period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    kama_vals = _compute_kama(closes, period, fast, slow)
    if not kama_vals:
        return None

    er_vals = _compute_er_series(closes, period)
    if not er_vals:
        return None

    cur_kama = kama_vals[-1]
    cur_er   = er_vals[-1]

    # Smoothing constant at current bar
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (cur_er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Price vs KAMA
    price_above = current > cur_kama
    dist_pct = (current - cur_kama) / cur_kama * 100.0 if cur_kama > 1e-9 else 0.0

    # KAMA direction over last 5 bars
    if len(kama_vals) >= 5:
        kama_rising = kama_vals[-1] > kama_vals[-2]
        kama_start  = kama_vals[-5]
        kama_slope_pct = (kama_vals[-1] - kama_start) / kama_start * 100.0 if kama_start > 1e-9 else 0.0
    else:
        kama_rising = True
        kama_slope_pct = 0.0

    # Market regime from ER
    if cur_er > 0.6:
        er_regime = "trending"
    elif cur_er < 0.3:
        er_regime = "choppy"
    else:
        er_regime = "transitional"

    # Score
    price_score = max(-100.0, min(100.0, dist_pct * 10.0))  # 10% above = +100
    dir_score = 30.0 if kama_rising else -30.0
    er_score = (cur_er - 0.5) * 60.0  # ER=1 → +30, ER=0 → -30
    score = price_score * 0.5 + dir_score * 0.3 + er_score * 0.2
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

    hist_kama = kama_vals[-history:]
    hist_er   = er_vals[-history:]

    verdict = (
        f"KAMA ({symbol}, {period}/{fast}/{slow}): {signal.replace('_', ' ')}. "
        f"KAMA={cur_kama:.2f}, ER={cur_er:.3f} ({er_regime}), "
        f"price {'+' if price_above else '-'}{abs(dist_pct):.2f}% from KAMA."
    )

    return KAMAReport(
        symbol=symbol,
        kama=round(cur_kama, 4),
        efficiency_ratio=round(cur_er, 4),
        smoothing_constant=round(sc, 6),
        price_above_kama=price_above,
        price_distance_pct=round(dist_pct, 3),
        kama_rising=kama_rising,
        kama_slope_pct=round(kama_slope_pct, 4),
        er_regime=er_regime,
        score=round(score, 2),
        signal=signal,
        kama_series=[round(v, 4) for v in hist_kama],
        er_series=[round(v, 4) for v in hist_er],
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
