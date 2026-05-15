"""Swing High and Low detection.

Identifies significant pivot points in price action:
  - Swing High: a bar whose high is higher than the surrounding N bars
  - Swing Low:  a bar whose low is lower than the surrounding N bars

Uses a "window" approach: bar[i] is a swing high if it has the highest high
in the range [i-window, i+window]. This avoids lookahead.

Applications:
  - Stop placement (stop below last swing low)
  - Target setting (target at next swing high)
  - Trend identification (higher highs + higher lows = uptrend)
  - Breakout detection (price exceeds last swing high)
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class SwingPoint:
    idx: int            # index in the bar series
    price: float        # swing high or low price
    kind: str           # "high" | "low"
    bar_date: str       # timestamp of the bar


@dataclass(frozen=True)
class SwingAnalysis:
    symbol: str
    current_price: float
    last_swing_high: SwingPoint | None
    last_swing_low: SwingPoint | None
    trend: str              # "uptrend" | "downtrend" | "sideways" | "unknown"
    stop_below: float | None   # suggested stop below last swing low
    target_above: float | None # suggested target at last swing high or next logical level
    breakout_up: bool       # current price above last swing high → breakout
    breakdown_down: bool    # current price below last swing low → breakdown
    swings_high: list[SwingPoint]  # all detected swing highs
    swings_low: list[SwingPoint]   # all detected swing lows


def detect_swings(
    bars: list[Bar],
    *,
    window: int = 3,
    lookback: int = 60,
) -> SwingAnalysis | None:
    """Detect swing highs and lows.

    window: bars on each side to confirm a pivot
    lookback: max bars to analyze
    Returns None if insufficient data.
    """
    if len(bars) < window * 2 + 2:
        return None

    symbol = bars[0].symbol
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    n = len(recent)
    current_price = recent[-1].close

    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []

    for i in range(window, n - window):
        bar = recent[i]
        # Swing high: this bar's high is >= all surrounding bars
        if all(bar.high >= recent[j].high for j in range(i - window, i + window + 1) if j != i):
            highs.append(SwingPoint(idx=i, price=bar.high, kind="high", bar_date=str(bar.ts)))
        # Swing low: this bar's low is <= all surrounding bars
        if all(bar.low <= recent[j].low for j in range(i - window, i + window + 1) if j != i):
            lows.append(SwingPoint(idx=i, price=bar.low, kind="low", bar_date=str(bar.ts)))

    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None

    # Trend classification using last 2 swing highs and lows
    trend = _classify_trend(highs, lows)

    # Stop suggestion: 0.5% below last swing low
    stop_below = None
    if last_low:
        stop_below = round(last_low.price * 0.995, 4)

    # Target suggestion: last swing high (if below current, use next)
    target_above = None
    if last_high:
        target_above = last_high.price
        # If we're already above the last swing high, project next target
        if current_price > last_high.price and len(highs) >= 2:
            # Use the distance between the last two swing highs as projection
            gap = highs[-1].price - highs[-2].price
            target_above = round(highs[-1].price + gap, 4)

    breakout_up = last_high is not None and current_price > last_high.price
    breakdown_down = last_low is not None and current_price < last_low.price

    return SwingAnalysis(
        symbol=symbol,
        current_price=round(current_price, 4),
        last_swing_high=last_high,
        last_swing_low=last_low,
        trend=trend,
        stop_below=stop_below,
        target_above=round(target_above, 4) if target_above else None,
        breakout_up=breakout_up,
        breakdown_down=breakdown_down,
        swings_high=highs,
        swings_low=lows,
    )


def _classify_trend(highs: list[SwingPoint], lows: list[SwingPoint]) -> str:
    """Classify trend from the last 2 swing highs and lows."""
    if len(highs) < 2 or len(lows) < 2:
        return "unknown"

    hh = highs[-1].price > highs[-2].price   # higher high
    hl = lows[-1].price > lows[-2].price      # higher low
    lh = highs[-1].price < highs[-2].price    # lower high
    ll = lows[-1].price < lows[-2].price      # lower low

    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    if hh and ll or lh and hl:
        return "sideways"
    return "unknown"
