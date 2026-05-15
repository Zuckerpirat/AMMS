"""On-Balance Volume (OBV).

OBV is a cumulative volume indicator:
  If close > prev_close: OBV += volume
  If close < prev_close: OBV -= volume
  If close = prev_close: OBV unchanged

Rising OBV with rising price → confirms uptrend (volume supporting move).
Falling OBV with rising price → bearish divergence (distribution).
Rising OBV with falling price → bullish divergence (accumulation).
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class OBVResult:
    obv: float          # Current cumulative OBV
    obv_ema: float      # EMA(20) of OBV for smoothed trend
    trend: str          # "rising" | "falling" | "flat"
    divergence: str     # "bullish" | "bearish" | "none"  (OBV vs price direction)


def obv(bars: list[Bar], ema_period: int = 20) -> OBVResult | None:
    """Compute OBV with EMA smoothing and divergence detection.

    Requires at least 2 bars. EMA requires ema_period + 1 bars for full signal.
    """
    if len(bars) < 2:
        return None

    cumulative = 0.0
    obv_series: list[float] = [0.0]

    for i in range(1, len(bars)):
        if bars[i].close > bars[i - 1].close:
            cumulative += bars[i].volume
        elif bars[i].close < bars[i - 1].close:
            cumulative -= bars[i].volume
        obv_series.append(cumulative)

    current_obv = obv_series[-1]

    # EMA of OBV series
    obv_ema = current_obv
    if len(obv_series) >= ema_period:
        k = 2.0 / (ema_period + 1)
        ema_val = sum(obv_series[:ema_period]) / ema_period
        for v in obv_series[ema_period:]:
            ema_val = v * k + ema_val * (1 - k)
        obv_ema = ema_val

    # Trend: compare last 5 OBV values
    if len(obv_series) >= 5:
        recent = obv_series[-5:]
        if recent[-1] > recent[0]:
            trend = "rising"
        elif recent[-1] < recent[0]:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"

    # Divergence: compare OBV trend vs price trend over last ~10 bars
    divergence = "none"
    if len(bars) >= 10 and len(obv_series) >= 10:
        price_change = bars[-1].close - bars[-10].close
        obv_change = obv_series[-1] - obv_series[-10]
        if price_change < 0 and obv_change > 0:
            divergence = "bullish"   # price falling, OBV rising → accumulation
        elif price_change > 0 and obv_change < 0:
            divergence = "bearish"   # price rising, OBV falling → distribution

    return OBVResult(
        obv=round(current_obv, 0),
        obv_ema=round(obv_ema, 0),
        trend=trend,
        divergence=divergence,
    )
