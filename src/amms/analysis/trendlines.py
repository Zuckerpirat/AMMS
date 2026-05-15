"""Automatic trend line detection.

Finds the dominant support and resistance trend lines from recent bars
by detecting swing pivot highs/lows and fitting lines through them.

Support line: drawn through the lowest pivot lows (uptrend floor)
Resistance line: drawn through the highest pivot highs (ceiling)

A trend line is defined by:
  - slope: price change per bar
  - intercept: price at bar 0
  - strength: R² of fit (1.0 = perfect line)
  - touches: number of pivot points on the line
  - breakout: whether current price has broken the line

Interpretation:
  - Rising support + falling resistance → converging wedge
  - Price near support → potential bounce
  - Price near resistance → potential reversal or breakout
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class TrendLine:
    slope: float           # price change per bar (positive = rising)
    intercept: float       # price at first bar of the series
    touches: int           # pivot points that fit this line
    strength: float        # R² of fit, 0..1
    current_value: float   # line price at current bar index
    direction: str         # "rising" | "falling" | "flat"


@dataclass(frozen=True)
class TrendLineResult:
    symbol: str
    current_price: float
    support: TrendLine | None
    resistance: TrendLine | None
    support_distance_pct: float | None    # % above current support line
    resistance_distance_pct: float | None  # % below current resistance line
    pattern: str   # "uptrend" | "downtrend" | "ranging" | "wedge" | "unknown"
    bars_used: int


def detect_trendlines(bars: list[Bar], pivot_window: int = 3, lookback: int = 60) -> TrendLineResult | None:
    """Detect support and resistance trend lines.

    pivot_window: bars on each side needed to confirm a pivot
    lookback: max bars to analyze
    Returns None if insufficient data.
    """
    if len(bars) < pivot_window * 2 + 3:
        return None

    symbol = bars[0].symbol
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    n = len(recent)

    # Find pivot highs and lows
    pivot_highs: list[tuple[int, float]] = []  # (index, price)
    pivot_lows: list[tuple[int, float]] = []

    for i in range(pivot_window, n - pivot_window):
        high = recent[i].high
        low = recent[i].low
        if all(recent[i].high >= recent[j].high for j in range(i - pivot_window, i + pivot_window + 1) if j != i):
            pivot_highs.append((i, high))
        if all(recent[i].low <= recent[j].low for j in range(i - pivot_window, i + pivot_window + 1) if j != i):
            pivot_lows.append((i, low))

    current_price = recent[-1].close
    current_idx = n - 1

    support = _fit_line(pivot_lows, current_idx) if len(pivot_lows) >= 2 else None
    resistance = _fit_line(pivot_highs, current_idx) if len(pivot_highs) >= 2 else None

    support_dist = None
    resistance_dist = None

    if support is not None and support.current_value > 0:
        support_dist = (current_price - support.current_value) / support.current_value * 100
    if resistance is not None and resistance.current_value > 0:
        resistance_dist = (resistance.current_value - current_price) / resistance.current_value * 100

    pattern = _classify_pattern(support, resistance)

    return TrendLineResult(
        symbol=symbol,
        current_price=round(current_price, 4),
        support=support,
        resistance=resistance,
        support_distance_pct=round(support_dist, 2) if support_dist is not None else None,
        resistance_distance_pct=round(resistance_dist, 2) if resistance_dist is not None else None,
        pattern=pattern,
        bars_used=n,
    )


def _fit_line(pivots: list[tuple[int, float]], current_idx: int) -> TrendLine | None:
    """Fit a linear trend line through pivot points (least squares)."""
    if len(pivots) < 2:
        return None

    xs = [float(p[0]) for p in pivots]
    ys = [p[1] for p in pivots]
    n = len(xs)

    mx = sum(xs) / n
    my = sum(ys) / n

    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_xx = sum((x - mx) ** 2 for x in xs)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx
    intercept = my - slope * mx

    # R² for goodness of fit
    y_pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    r2 = max(0.0, min(1.0, r2))

    current_value = slope * current_idx + intercept

    if slope > 0.01:
        direction = "rising"
    elif slope < -0.01:
        direction = "falling"
    else:
        direction = "flat"

    return TrendLine(
        slope=round(slope, 4),
        intercept=round(intercept, 4),
        touches=n,
        strength=round(r2, 3),
        current_value=round(current_value, 4),
        direction=direction,
    )


def _classify_pattern(support: TrendLine | None, resistance: TrendLine | None) -> str:
    if support is None and resistance is None:
        return "unknown"
    if support is None:
        return "downtrend" if resistance and resistance.direction == "falling" else "unknown"
    if resistance is None:
        return "uptrend" if support.direction == "rising" else "unknown"

    if support.direction == "rising" and resistance.direction == "rising":
        return "uptrend"
    if support.direction == "falling" and resistance.direction == "falling":
        return "downtrend"
    if support.direction == "rising" and resistance.direction == "falling":
        return "wedge"
    return "ranging"
