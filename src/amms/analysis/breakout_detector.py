"""Breakout detection.

Identifies potential breakouts from consolidation phases by detecting:
  1. Price range compression (Bollinger Band squeeze or ATR compression)
  2. Volume expansion on the breakout bar
  3. Direction of the potential breakout

Signal types:
  - "breakout_up":   price broke above recent resistance with volume
  - "breakout_down": price broke below recent support with volume
  - "squeeze":       range compression without breakout yet (pre-breakout)
  - "none":          no breakout signal

Confidence: 0-100 based on strength of the signal components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BreakoutSignal:
    symbol: str
    signal: str              # "breakout_up"|"breakout_down"|"squeeze"|"none"
    confidence: float        # 0-100
    current_price: float
    breakout_level: float | None   # price level that was broken
    volume_ratio: float            # current vol vs avg (1.0 = average)
    range_compression: float       # current ATR / avg ATR (1.0 = average)
    is_squeeze: bool               # True if range compressed
    bars_used: int


def detect(
    bars: list,
    *,
    lookback: int = 20,
    squeeze_threshold: float = 0.7,
    vol_expansion_threshold: float = 1.3,
) -> BreakoutSignal | None:
    """Detect breakout signals.

    bars: list[Bar] — needs at least lookback + 5 bars
    lookback: window for baseline calculations (default 20)
    squeeze_threshold: ATR compression ratio to call a squeeze (default 0.7 = 30% below avg)
    vol_expansion_threshold: min volume ratio to confirm breakout (default 1.3)
    Returns None if insufficient data.
    """
    if len(bars) < lookback + 5:
        return None

    symbol = bars[0].symbol

    # Full window for baseline, last bar for current
    baseline = bars[-(lookback + 5): -1]   # previous bars for average
    n_base = len(baseline)
    current_bar = bars[-1]
    current_price = current_bar.close

    # ATR for baseline and current
    def bar_tr(b, prev_b) -> float:
        return max(
            b.high - b.low,
            abs(b.high - prev_b.close),
            abs(b.low - prev_b.close),
        )

    base_trs = [bar_tr(baseline[i], baseline[i - 1]) for i in range(1, n_base)]
    base_trs += [bar_tr(current_bar, bars[-2])]

    avg_atr = sum(base_trs[:-1]) / max(1, len(base_trs) - 1)
    current_atr = base_trs[-1]

    range_compression = current_atr / avg_atr if avg_atr > 0 else 1.0
    is_squeeze = range_compression < squeeze_threshold

    # Volume ratio
    avg_vol = sum(b.volume for b in baseline) / n_base if n_base > 0 else 1.0
    current_vol = current_bar.volume
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Recent range (highest high, lowest low over lookback)
    window = bars[-lookback - 1: -1]
    recent_high = max(b.high for b in window) if window else current_price
    recent_low = min(b.low for b in window) if window else current_price

    # Breakout detection
    broke_up = current_price > recent_high
    broke_down = current_price < recent_low
    vol_confirms = volume_ratio >= vol_expansion_threshold

    if broke_up and vol_confirms:
        signal = "breakout_up"
        breakout_level = recent_high
        confidence = min(100.0, 50 + volume_ratio * 15 + (1 - range_compression) * 20)
    elif broke_down and vol_confirms:
        signal = "breakout_down"
        breakout_level = recent_low
        confidence = min(100.0, 50 + volume_ratio * 15 + (1 - range_compression) * 20)
    elif is_squeeze:
        signal = "squeeze"
        breakout_level = None
        confidence = min(100.0, (1 - range_compression) * 100)
    else:
        signal = "none"
        breakout_level = None
        confidence = 0.0

    return BreakoutSignal(
        symbol=symbol,
        signal=signal,
        confidence=round(confidence, 1),
        current_price=round(current_price, 4),
        breakout_level=round(breakout_level, 4) if breakout_level else None,
        volume_ratio=round(volume_ratio, 2),
        range_compression=round(range_compression, 3),
        is_squeeze=is_squeeze,
        bars_used=len(bars),
    )
