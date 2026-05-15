"""Average Directional Index (ADX) — trend strength indicator.

ADX measures trend strength regardless of direction (0..100).
+DI and -DI indicate bullish vs bearish directional pressure.

Interpretation:
  ADX < 20  : weak/no trend (range-bound, choppy)
  ADX 20-25 : emerging trend
  ADX 25-40 : strong trend
  ADX > 40  : very strong trend
  ADX > 50  : extreme trend (often near exhaustion)

+DI > -DI  : bullish
+DI < -DI  : bearish
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class ADXResult:
    adx: float          # 0..100 — trend strength
    plus_di: float      # +DI — bullish directional indicator
    minus_di: float     # -DI — bearish directional indicator
    trend_strength: str  # "none" | "emerging" | "strong" | "very_strong" | "extreme"
    direction: str       # "bullish" | "bearish" | "neutral"


def adx(bars: list[Bar], period: int = 14) -> ADXResult | None:
    """Compute ADX, +DI, and -DI for the given bars.

    Requires at least 2*period + 1 bars for a stable result.
    Returns None if insufficient data.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if len(bars) < period * 2 + 1:
        return None

    # Step 1: compute True Range, +DM, -DM for each bar (starting from bar[1])
    tr_list: list[float] = []
    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []

    for i in range(1, len(bars)):
        prev = bars[i - 1]
        curr = bars[i]
        high_diff = curr.high - prev.high
        low_diff = prev.low - curr.low

        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
        )
        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0.0

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Step 2: smoothed sums (Wilder smoothing — TR/DM use running sum seed)
    def _wilder_sum(values: list[float], n: int) -> list[float]:
        """Wilder smooth for TR and DM: seed = first n sum, add raw values."""
        if len(values) < n:
            return []
        smoothed = [sum(values[:n])]
        for v in values[n:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / n + v)
        return smoothed

    def _wilder_ema(values: list[float], n: int) -> list[float]:
        """Wilder EMA for DX→ADX: seed = mean of first n, update adds v/n."""
        if len(values) < n:
            return []
        seed = sum(values[:n]) / n
        smoothed = [seed]
        for v in values[n:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / n + v / n)
        return smoothed

    atr_smooth = _wilder_sum(tr_list, period)
    plus_dm_smooth = _wilder_sum(plus_dm_list, period)
    minus_dm_smooth = _wilder_sum(minus_dm_list, period)

    if not atr_smooth:
        return None

    # Step 3: DI lines
    plus_di_series = [
        100.0 * pdm / atr if atr > 0 else 0.0
        for pdm, atr in zip(plus_dm_smooth, atr_smooth)
    ]
    minus_di_series = [
        100.0 * mdm / atr if atr > 0 else 0.0
        for mdm, atr in zip(minus_dm_smooth, atr_smooth)
    ]

    # Step 4: DX and ADX (Wilder smooth of DX)
    dx_list: list[float] = []
    for pdi, mdi in zip(plus_di_series, minus_di_series):
        di_sum = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0
        dx_list.append(dx)

    adx_smooth = _wilder_ema(dx_list, period)
    if not adx_smooth:
        return None

    adx_val = adx_smooth[-1]
    plus_di = plus_di_series[-1]
    minus_di = minus_di_series[-1]

    if adx_val < 20:
        strength = "none"
    elif adx_val < 25:
        strength = "emerging"
    elif adx_val < 40:
        strength = "strong"
    elif adx_val < 50:
        strength = "very_strong"
    else:
        strength = "extreme"

    if plus_di > minus_di + 2:
        direction = "bullish"
    elif minus_di > plus_di + 2:
        direction = "bearish"
    else:
        direction = "neutral"

    return ADXResult(
        adx=round(adx_val, 2),
        plus_di=round(plus_di, 2),
        minus_di=round(minus_di, 2),
        trend_strength=strength,
        direction=direction,
    )
