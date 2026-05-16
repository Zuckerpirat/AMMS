"""Price compression and volatility squeeze detector.

Identifies when price is compressing into a tight range as a precursor
to a breakout. Uses two signals:

1. ATR Compression: current ATR is significantly lower than its N-period
   average (ATR contracting = lower volatility = potential energy building).

2. Bollinger Band Width: measures band width relative to its own history.
   Very narrow bands (BB squeeze) often precede large moves.

Also computes:
  - Days/bars since ATR was at its 4-week low (squeeze duration)
  - Breakout bias hint: compares recent close vs midpoint of range

When both signals fire simultaneously → high-probability squeeze setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionReport:
    symbol: str
    atr_current: float
    atr_average: float
    atr_ratio: float          # current / average — <0.7 = compressed
    bb_width_current: float   # (upper - lower) / middle, %
    bb_width_average: float
    bb_width_ratio: float     # current / average — <0.7 = compressed
    compressed: bool          # True when both signals fire
    compression_strength: float   # 0-100 (100 = maximum compression)
    bars_in_squeeze: int      # how many consecutive bars in compression
    bias: str                 # "bullish" / "bearish" / "neutral" (price position)
    current_price: float
    range_high: float         # N-bar range high
    range_low: float          # N-bar range low
    bars_used: int
    verdict: str


def _atr(bars: list, period: int = 14) -> list[float]:
    """Returns list of ATR values, one per bar from index period onward."""
    if len(bars) < period + 1:
        return []
    trs = []
    for i in range(1, len(bars)):
        high = float(bars[i].high)
        low = float(bars[i].low)
        prev_close = float(bars[i - 1].close)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atrs = []
    if len(trs) < period:
        return []
    # Wilder smoothing
    wilder = sum(trs[:period]) / period
    atrs.append(wilder)
    for tr in trs[period:]:
        wilder = (wilder * (period - 1) + tr) / period
        atrs.append(wilder)
    return atrs


def _bb_width(closes: list[float], period: int = 20, n_std: float = 2.0) -> list[float]:
    """Returns list of Bollinger Band width values (%)."""
    widths = []
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mid = sum(window) / period
        std = math.sqrt(sum((c - mid) ** 2 for c in window) / period)
        width = (4 * n_std * std / mid * 100) if mid > 0 else 0.0
        widths.append(width)
    return widths


def analyze(bars: list, *, symbol: str = "", lookback: int = 20, atr_threshold: float = 0.75, bb_threshold: float = 0.75) -> CompressionReport | None:
    """Detect price compression / volatility squeeze.

    bars: list[Bar] with .high .low .close — at least 35 bars.
    symbol: ticker for display.
    lookback: window for averaging ATR and BB width.
    atr_threshold: ATR ratio below which compression is flagged.
    bb_threshold: BB width ratio below which compression is flagged.
    """
    if not bars or len(bars) < max(35, lookback + 15):
        return None

    try:
        closes = [float(b.close) for b in bars]
        current_price = closes[-1]
    except Exception:
        return None

    atrs = _atr(bars)
    if not atrs or len(atrs) < lookback:
        return None

    atr_current = atrs[-1]
    # Use longer baseline (2× lookback) for average to detect recent compression
    baseline = min(len(atrs), lookback * 2)
    atr_avg = sum(atrs[-baseline:]) / baseline
    atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0

    bbws = _bb_width(closes)
    if not bbws or len(bbws) < lookback:
        return None

    bb_current = bbws[-1]
    # Use all available BB width history for the baseline average
    bb_baseline = min(len(bbws), lookback * 2)
    bb_avg = sum(bbws[-bb_baseline:]) / bb_baseline
    bb_ratio = bb_current / bb_avg if bb_avg > 0 else 1.0

    atr_compressed = atr_ratio < atr_threshold
    bb_compressed = bb_ratio < bb_threshold
    compressed = atr_compressed and bb_compressed

    # Compression strength: 0-100 (lower ratio = stronger)
    atr_strength = max(0.0, (1 - atr_ratio) / (1 - atr_threshold + 0.01)) * 50
    bb_strength = max(0.0, (1 - bb_ratio) / (1 - bb_threshold + 0.01)) * 50
    strength = min(100.0, atr_strength + bb_strength)

    # Bars in squeeze (consecutive bars where atr_ratio < threshold)
    squeeze_bars = 0
    for atr_val in reversed(atrs):
        if atr_val < atr_avg * atr_threshold:
            squeeze_bars += 1
        else:
            break

    # Range and bias
    range_bars = bars[-lookback:]
    range_high = max(float(b.high) for b in range_bars)
    range_low = min(float(b.low) for b in range_bars)
    range_mid = (range_high + range_low) / 2
    if current_price > range_mid * 1.01:
        bias = "bullish"
    elif current_price < range_mid * 0.99:
        bias = "bearish"
    else:
        bias = "neutral"

    if compressed:
        verdict = (
            f"⚡ SQUEEZE DETECTED — ATR at {atr_ratio:.0%} of avg, "
            f"BB width at {bb_ratio:.0%} of avg. "
            f"Compression {squeeze_bars} bars. "
            f"Bias: {bias}. Expect volatility expansion soon."
        )
    elif atr_compressed or bb_compressed:
        verdict = (
            f"Partial compression — "
            f"ATR {atr_ratio:.0%} of avg, BB width {bb_ratio:.0%} of avg. "
            f"Monitor for developing squeeze."
        )
    else:
        verdict = (
            f"No compression — volatility normal. "
            f"ATR {atr_ratio:.0%} of avg, BB width {bb_ratio:.0%} of avg."
        )

    return CompressionReport(
        symbol=symbol,
        atr_current=round(atr_current, 4),
        atr_average=round(atr_avg, 4),
        atr_ratio=round(atr_ratio, 3),
        bb_width_current=round(bb_current, 3),
        bb_width_average=round(bb_avg, 3),
        bb_width_ratio=round(bb_ratio, 3),
        compressed=compressed,
        compression_strength=round(strength, 1),
        bars_in_squeeze=squeeze_bars,
        bias=bias,
        current_price=round(current_price, 2),
        range_high=round(range_high, 2),
        range_low=round(range_low, 2),
        bars_used=len(bars),
        verdict=verdict,
    )
