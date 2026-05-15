"""Fibonacci Retracement and Extension levels.

Given a swing high and swing low, computes the key Fibonacci levels:

Retracement levels (measured from the move):
  0%    = swing low (start of move)
  23.6% = shallow retracement
  38.2% = moderate retracement
  50.0% = half the move (not Fibonacci, but widely used)
  61.8% = golden ratio retracement (strongest)
  78.6% = deep retracement
  100%  = swing high (full retracement = trend reversal)

Extension levels (for targets beyond the swing high):
  127.2%, 161.8%, 200%, 261.8%

Auto-detection: finds the most recent significant swing high and low
within a lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
EXTENSION_LEVELS = [1.272, 1.618, 2.0, 2.618]
ALL_LEVELS = RETRACEMENT_LEVELS + EXTENSION_LEVELS


@dataclass(frozen=True)
class FibLevel:
    ratio: float
    price: float
    label: str    # e.g. "61.8%" or "Ext 161.8%"
    is_extension: bool


@dataclass(frozen=True)
class FibResult:
    swing_low: float
    swing_high: float
    current_price: float
    direction: str   # "uptrend" (low→high) | "downtrend" (high→low)
    levels: list[FibLevel]
    nearest_support: FibLevel | None   # closest level below current price
    nearest_resistance: FibLevel | None  # closest level above current price


def fibonacci(bars: list[Bar], lookback: int = 50) -> FibResult | None:
    """Compute Fibonacci levels from most recent swing high/low.

    Automatically detects the swing high and low within `lookback` bars.
    Returns None if insufficient data.
    """
    if len(bars) < 5:
        return None

    recent = bars[-lookback:] if len(bars) >= lookback else bars
    swing_low = min(b.low for b in recent)
    swing_high = max(b.high for b in recent)

    if swing_high <= swing_low:
        return None

    price = bars[-1].close

    # Determine direction: was the high or low more recent?
    high_idx = max(range(len(recent)), key=lambda i: recent[i].high)
    low_idx = min(range(len(recent)), key=lambda i: recent[i].low)

    if high_idx > low_idx:
        direction = "uptrend"   # low came first, then high
        anchor_low = swing_low
        anchor_high = swing_high
    else:
        direction = "downtrend"  # high came first, then low
        anchor_low = swing_low
        anchor_high = swing_high

    move = anchor_high - anchor_low
    levels: list[FibLevel] = []

    for ratio in RETRACEMENT_LEVELS:
        if direction == "uptrend":
            level_price = anchor_high - ratio * move  # retracement from high
        else:
            level_price = anchor_low + ratio * move   # recovery from low
        pct = ratio * 100
        label = f"{pct:.1f}%"
        levels.append(FibLevel(ratio=ratio, price=round(level_price, 4), label=label, is_extension=False))

    for ratio in EXTENSION_LEVELS:
        if direction == "uptrend":
            level_price = anchor_low + ratio * move  # extension above high
        else:
            level_price = anchor_high - ratio * move
        pct = ratio * 100
        label = f"Ext {pct:.1f}%"
        levels.append(FibLevel(ratio=ratio, price=round(level_price, 4), label=label, is_extension=True))

    # Sort by price ascending
    levels.sort(key=lambda x: x.price)

    # Find nearest support (closest level below price)
    support = max((lv for lv in levels if lv.price < price), key=lambda x: x.price, default=None)
    resistance = min((lv for lv in levels if lv.price > price), key=lambda x: x.price, default=None)

    return FibResult(
        swing_low=round(swing_low, 4),
        swing_high=round(swing_high, 4),
        current_price=round(price, 4),
        direction=direction,
        levels=levels,
        nearest_support=support,
        nearest_resistance=resistance,
    )
