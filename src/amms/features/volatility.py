from __future__ import annotations

import math

from amms.data.bars import Bar

TRADING_DAYS_PER_YEAR = 252


def atr(bars: list[Bar], n: int = 14) -> float | None:
    """Average True Range over ``n`` bars. Needs at least n+1 bars of history."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n + 1:
        return None
    trs: list[float] = []
    for i in range(len(bars) - n, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close
        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(true_range)
    return sum(trs) / n


def realized_vol(bars: list[Bar], n: int = 20) -> float | None:
    """Annualized realized volatility from daily log returns over ``n`` bars.

    Returns None when fewer than n+1 bars are available or when prices fall
    to zero (log return undefined). Result is sqrt(252) * stddev(log returns).
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n + 1:
        return None
    log_returns: list[float] = []
    for i in range(len(bars) - n, len(bars)):
        prev = bars[i - 1].close
        curr = bars[i].close
        if prev <= 0 or curr <= 0:
            return None
        log_returns.append(math.log(curr / prev))
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)
