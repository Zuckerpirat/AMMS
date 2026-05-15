"""Bollinger Bands.

Upper band = SMA(n) + k * std(n)
Lower band = SMA(n) - k * std(n)
%B = (price - lower) / (upper - lower)   [0=at lower, 1=at upper, 0.5=at mean]
Bandwidth = (upper - lower) / SMA(n)      [higher = wider bands, more volatile]
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class BollingerBands:
    upper: float
    middle: float  # SMA
    lower: float
    pct_b: float     # 0..1 (0 = at lower, 1 = at upper)
    bandwidth: float  # (upper-lower)/middle — relative band width


def bollinger(bars: list[Bar], n: int = 20, k: float = 2.0) -> BollingerBands | None:
    """Compute Bollinger Bands for the last ``n`` bars.

    Returns None if fewer than n bars or zero std dev.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if len(bars) < n:
        return None
    closes = [b.close for b in bars[-n:]]
    mean = statistics.fmean(closes)
    std = statistics.stdev(closes)
    if std == 0:
        return BollingerBands(upper=mean, middle=mean, lower=mean, pct_b=0.5, bandwidth=0.0)
    upper = mean + k * std
    lower = mean - k * std
    price = closes[-1]
    band_width = upper - lower
    pct_b = (price - lower) / band_width if band_width > 0 else 0.5
    bandwidth = band_width / mean if mean > 0 else 0.0
    return BollingerBands(
        upper=round(upper, 4),
        middle=round(mean, 4),
        lower=round(lower, 4),
        pct_b=round(pct_b, 4),
        bandwidth=round(bandwidth, 4),
    )


def volume_spike(bars: list[Bar], n: int = 20, threshold: float = 2.0) -> float | None:
    """Ratio of last bar's volume to average volume over n bars.

    Returns None if insufficient data. ratio > threshold indicates a spike.
    """
    if len(bars) < n + 1:
        return None
    avg_vol = sum(b.volume for b in bars[-(n + 1):-1]) / n
    if avg_vol <= 0:
        return None
    return bars[-1].volume / avg_vol
