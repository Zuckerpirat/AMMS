"""Pairs trading spread analysis.

Computes the price spread and statistical relationship between two assets.
Used to identify mean-reversion opportunities when correlated assets diverge.

Metrics:
  - Price ratio (sym1/sym2) and its Z-score
  - Correlation coefficient (30-day returns)
  - Cointegration signal (spread deviation from mean)
  - Trading signal: enter when spread Z-score > threshold

A Z-score > 2 suggests the spread is unusually wide → mean reversion expected.
A Z-score < -2 suggests the spread is unusually narrow → also a trade signal.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class PairsResult:
    sym1: str
    sym2: str
    current_ratio: float        # current price of sym1 / sym2
    ratio_mean: float           # rolling mean of ratio
    ratio_std: float            # rolling std of ratio
    ratio_zscore: float         # (current_ratio - mean) / std
    correlation: float          # pearson correlation of returns
    signal: str                 # "long_spread" | "short_spread" | "neutral"
    signal_strength: float      # 0..1
    recommended_action: str


def analyze_pair(
    bars1: list[Bar],
    bars2: list[Bar],
    *,
    n: int = 30,
) -> PairsResult | None:
    """Analyze the price spread between two assets.

    bars1, bars2: bar series for sym1 and sym2 (must have same length or be aligned by date)
    n: lookback period for ratio statistics
    """
    if len(bars1) < n + 1 or len(bars2) < n + 1:
        return None

    sym1 = bars1[0].symbol
    sym2 = bars2[0].symbol

    # Use the last n+1 bars (n+1 for n returns)
    b1 = bars1[-(n + 1):]
    b2 = bars2[-(n + 1):]

    closes1 = [b.close for b in b1]
    closes2 = [b.close for b in b2]

    # Price ratio series
    ratios = [c1 / c2 if c2 > 0 else None for c1, c2 in zip(closes1, closes2)]
    valid_ratios = [r for r in ratios if r is not None]

    if len(valid_ratios) < n:
        return None

    current_ratio = valid_ratios[-1]
    ratio_mean = statistics.fmean(valid_ratios)
    ratio_std = statistics.stdev(valid_ratios)

    if ratio_std == 0:
        ratio_zscore = 0.0
    else:
        ratio_zscore = (current_ratio - ratio_mean) / ratio_std

    # Pearson correlation of daily returns
    returns1 = [(closes1[i] - closes1[i - 1]) / closes1[i - 1] for i in range(1, len(closes1)) if closes1[i - 1] > 0]
    returns2 = [(closes2[i] - closes2[i - 1]) / closes2[i - 1] for i in range(1, len(closes2)) if closes2[i - 1] > 0]

    correlation = _pearson(returns1, returns2)

    # Trading signal
    threshold = 1.5
    if ratio_zscore > 2.0:
        signal = "long_spread"  # sym1 relatively expensive vs sym2 → expected to mean-revert
        strength = min((ratio_zscore - 2.0) / 2.0 + 0.5, 1.0)
        action = (f"Spread wide: {sym1} overbought vs {sym2}. "
                  f"Mean-reversion: sell {sym1} / buy {sym2}.")
    elif ratio_zscore < -2.0:
        signal = "short_spread"  # sym1 relatively cheap vs sym2
        strength = min((-ratio_zscore - 2.0) / 2.0 + 0.5, 1.0)
        action = (f"Spread narrow: {sym1} oversold vs {sym2}. "
                  f"Mean-reversion: buy {sym1} / sell {sym2}.")
    elif abs(ratio_zscore) > threshold:
        signal = "long_spread" if ratio_zscore > 0 else "short_spread"
        strength = abs(ratio_zscore) / 3.0
        action = f"Mild spread deviation (z={ratio_zscore:+.2f}). Monitor."
    else:
        signal = "neutral"
        strength = 0.0
        action = "Spread within normal range. No trade signal."

    return PairsResult(
        sym1=sym1,
        sym2=sym2,
        current_ratio=round(current_ratio, 4),
        ratio_mean=round(ratio_mean, 4),
        ratio_std=round(ratio_std, 4),
        ratio_zscore=round(ratio_zscore, 3),
        correlation=round(correlation, 3),
        signal=signal,
        signal_strength=round(strength, 3),
        recommended_action=action,
    )


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da * db > 0 else 0.0
