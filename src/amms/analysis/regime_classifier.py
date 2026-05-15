"""Market regime classification.

Classifies the current market regime for a symbol based on:
  1. Trend strength (ADX proxy via directional movement)
  2. Volatility level (realized vol vs its own median)
  3. Momentum direction (net price change over lookback)

Four primary regimes:
  - "trending_up":    strong uptrend, clear directional movement
  - "trending_down":  strong downtrend, clear directional movement
  - "ranging_low_vol": sideways, compressed volatility (pre-breakout)
  - "ranging_high_vol": choppy/mean-reverting, elevated volatility

Each regime suggests a different strategy approach:
  - trending_up → momentum / trend-following
  - trending_down → short bias / defensive / cash
  - ranging_low_vol → breakout preparation, mean reversion entries
  - ranging_high_vol → reduce size, mean reversion only with tight stops

Confidence: 0-100 (how clearly the regime signal is defined)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeResult:
    symbol: str
    regime: str                   # see module docstring
    confidence: float             # 0-100
    trend_strength: float         # 0-100 (ADX-like)
    trend_direction: str          # "up"|"down"|"flat"
    vol_regime: str               # "low"|"normal"|"high"
    momentum_pct: float           # net % change over lookback
    strategy_hint: str            # brief strategy suggestion
    bars_used: int


def classify(bars: list, *, lookback: int = 20) -> RegimeResult | None:
    """Classify market regime for the given bars.

    bars: list[Bar] — needs at least lookback + 5 bars
    lookback: analysis window (default 20)
    Returns None if insufficient data.
    """
    if len(bars) < lookback + 5 or lookback < 5:
        return None

    symbol = bars[0].symbol
    window = bars[-lookback:]
    n = len(window)

    closes = [b.close for b in window]
    highs = [b.high for b in window]
    lows = [b.low for b in window]

    # Trend direction: linear regression slope
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n
    numerator = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denom if denom > 0 else 0.0
    slope_pct = slope / closes[0] * 100 if closes[0] > 0 else 0.0

    if slope_pct > 0.15:
        trend_direction = "up"
    elif slope_pct < -0.15:
        trend_direction = "down"
    else:
        trend_direction = "flat"

    # ADX proxy: directional movement
    plus_dm_sum = 0.0
    minus_dm_sum = 0.0
    tr_sum = 0.0

    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0.0
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        plus_dm_sum += plus_dm
        minus_dm_sum += minus_dm
        tr_sum += tr

    if tr_sum > 0:
        plus_di = plus_dm_sum / tr_sum * 100
        minus_di = minus_dm_sum / tr_sum * 100
        di_sum = plus_di + minus_di
        adx_proxy = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0.0
    else:
        adx_proxy = 0.0
        plus_di = minus_di = 0.0

    trend_strength = min(100.0, adx_proxy)

    # Realized volatility (annualized)
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, n) if closes[i - 1] > 0]
    if len(log_rets) >= 3:
        mean_ret = sum(log_rets) / len(log_rets)
        variance = sum((r - mean_ret) ** 2 for r in log_rets) / len(log_rets)
        realized_vol = math.sqrt(variance) * math.sqrt(252) * 100
    else:
        realized_vol = 0.0

    # Compare current vol to a rough "normal" baseline
    if realized_vol > 40:
        vol_regime = "high"
    elif realized_vol > 15:
        vol_regime = "normal"
    else:
        vol_regime = "low"

    # Net momentum
    momentum_pct = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0.0

    # Regime classification
    is_trending = trend_strength > 25
    is_up = trend_direction == "up"

    if is_trending and is_up:
        regime = "trending_up"
        confidence = min(100.0, trend_strength + (10 if vol_regime != "high" else 0))
        hint = "Momentum / trend-following. Hold winners. Trail stops."
    elif is_trending and not is_up:
        regime = "trending_down"
        confidence = min(100.0, trend_strength + (10 if vol_regime != "high" else 0))
        hint = "Defensive positioning. Reduce exposure. Cash is valid."
    elif vol_regime == "low":
        regime = "ranging_low_vol"
        confidence = min(100.0, 40 + (30 - trend_strength))
        hint = "Coiled spring. Await breakout. Mean reversion entries OK."
    else:
        regime = "ranging_high_vol"
        confidence = min(100.0, 50 + (30 - trend_strength))
        hint = "Choppy market. Reduce size. Mean reversion with tight stops."

    STRATEGY_HINTS = {
        "trending_up": "Momentum / trend-following. Hold winners. Trail stops.",
        "trending_down": "Defensive positioning. Reduce exposure. Cash is valid.",
        "ranging_low_vol": "Coiled spring. Await breakout. Mean reversion entries OK.",
        "ranging_high_vol": "Choppy market. Reduce size. Mean reversion with tight stops.",
    }

    return RegimeResult(
        symbol=symbol,
        regime=regime,
        confidence=round(confidence, 1),
        trend_strength=round(trend_strength, 1),
        trend_direction=trend_direction,
        vol_regime=vol_regime,
        momentum_pct=round(momentum_pct, 2),
        strategy_hint=STRATEGY_HINTS[regime],
        bars_used=n,
    )
