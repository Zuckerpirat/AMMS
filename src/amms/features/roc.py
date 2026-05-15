"""Rate of Change (ROC) / Momentum indicator.

ROC = (close_now - close_n_bars_ago) / close_n_bars_ago * 100

Measures the percentage change in price over N periods.
Unlike RSI (which normalizes) or MACD (which uses EMAs), ROC is a raw
measure of momentum — how much has price moved in N bars?

Interpretation:
  ROC > 0  → positive momentum (price higher than N bars ago)
  ROC < 0  → negative momentum (price lower than N bars ago)
  |ROC| increasing → momentum accelerating
  ROC crossing 0 → momentum reversal

Multiple periods (10, 20, 50) give short/medium/long-term momentum picture.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class ROCResult:
    value: float        # percentage rate of change
    period: int
    momentum: str       # "strong_up" | "up" | "flat" | "down" | "strong_down"


def roc(bars: list[Bar], period: int = 10) -> ROCResult | None:
    """Compute Rate of Change over period bars.

    Returns None if fewer than period+1 bars.
    """
    if len(bars) < period + 1:
        return None

    close_now = bars[-1].close
    close_then = bars[-(period + 1)].close

    if close_then <= 0:
        return None

    value = (close_now - close_then) / close_then * 100.0

    if value > 5.0:
        momentum = "strong_up"
    elif value > 1.0:
        momentum = "up"
    elif value < -5.0:
        momentum = "strong_down"
    elif value < -1.0:
        momentum = "down"
    else:
        momentum = "flat"

    return ROCResult(
        value=round(value, 2),
        period=period,
        momentum=momentum,
    )


@dataclass(frozen=True)
class MultiROC:
    short: ROCResult | None    # 10-bar ROC
    medium: ROCResult | None   # 20-bar ROC
    long: ROCResult | None     # 50-bar ROC
    overall: str               # "accelerating_up" | "decelerating" | "accelerating_down" | "mixed" | "flat"


def multi_roc(bars: list[Bar]) -> MultiROC:
    """Compute ROC at 10, 20, and 50 bars and classify the overall trend.

    Useful for comparing short-term vs long-term momentum alignment.
    """
    r10 = roc(bars, 10)
    r20 = roc(bars, 20)
    r50 = roc(bars, 50)

    vals = [r.value for r in [r10, r20, r50] if r is not None]

    if not vals:
        overall = "flat"
    elif all(v > 1.0 for v in vals):
        overall = "accelerating_up"
    elif all(v < -1.0 for v in vals):
        overall = "accelerating_down"
    elif any(v > 1.0 for v in vals) and any(v < -1.0 for v in vals):
        overall = "mixed"
    elif max(vals) - min(vals) < 1.0:
        overall = "flat"
    else:
        overall = "decelerating"

    return MultiROC(short=r10, medium=r20, long=r50, overall=overall)
