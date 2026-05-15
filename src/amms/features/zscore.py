"""Z-score indicator.

Measures how many standard deviations the current price is from its
rolling mean. Used for mean-reversion strategies.

  z = (price - mean(n)) / std(n)

  z < -2  : potentially oversold (price far below mean)
  z > +2  : potentially overbought (price far above mean)
  -1..+1  : normal range
"""

from __future__ import annotations

import statistics

from amms.data.bars import Bar


def zscore(bars: list[Bar], n: int = 20) -> float | None:
    """Z-score of the last bar's close vs the rolling n-bar mean/std.

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
        return 0.0
    return (closes[-1] - mean) / std


def zscore_series(bars: list[Bar], n: int = 20) -> list[float]:
    """Compute z-score for every bar that has n prior bars available.

    Returns a list with len = max(0, len(bars) - n + 1).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    result: list[float] = []
    for i in range(n - 1, len(bars)):
        window = bars[i - n + 1: i + 1]
        closes = [b.close for b in window]
        mean = statistics.fmean(closes)
        std = statistics.stdev(closes)
        if std == 0:
            result.append(0.0)
        else:
            result.append((closes[-1] - mean) / std)
    return result
