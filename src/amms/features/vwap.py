"""Volume-Weighted Average Price (VWAP) computation.

VWAP = sum(price * volume) / sum(volume)

Uses the typical price (high + low + close) / 3 per bar.
Can be computed over a rolling window of bars (not a true intraday VWAP
which resets each session, but a useful multi-day proxy).
"""

from __future__ import annotations

from amms.data.bars import Bar


def vwap(bars: list[Bar], n: int | None = None) -> float | None:
    """Compute VWAP over the last ``n`` bars (or all bars if n is None).

    Returns None when there are no bars or total volume is zero.
    """
    window = bars[-n:] if n is not None and n > 0 else bars
    if not window:
        return None
    total_vol = sum(b.volume for b in window)
    if total_vol <= 0:
        return None
    total_pv = sum(((b.high + b.low + b.close) / 3) * b.volume for b in window)
    return total_pv / total_vol


def vwap_deviation_pct(price: float, bars: list[Bar], n: int | None = None) -> float | None:
    """Percentage deviation of ``price`` from VWAP.

    Positive = price above VWAP (momentum), negative = below (discount).
    Returns None when VWAP cannot be computed.
    """
    v = vwap(bars, n)
    if v is None or v <= 0:
        return None
    return (price - v) / v * 100
