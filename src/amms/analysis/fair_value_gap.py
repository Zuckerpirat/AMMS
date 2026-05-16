"""Fair Value Gap (FVG) detector.

A Fair Value Gap is a three-candle pattern where price moves so fast
that it leaves an imbalance zone between candle 1's wick and candle 3's
opposite wick.

Bullish FVG (gap up): candle 3 low > candle 1 high
  → imbalance zone: [candle1.high, candle3.low]
  → price may return to fill this zone (institutional buying zone)

Bearish FVG (gap down): candle 3 high < candle 1 low
  → imbalance zone: [candle3.high, candle1.low]
  → price may return to fill this zone (institutional selling zone)

Partially filled FVGs (price entered the zone) are tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FairValueGap:
    kind: str         # "bullish" / "bearish"
    bar_index: int    # index of the middle candle
    upper: float      # top of gap zone
    lower: float      # bottom of gap zone
    midpoint: float
    size_pct: float   # gap size as % of price
    filled: bool      # True if current price has entered the zone
    partial: bool     # True if price partially entered the gap


@dataclass(frozen=True)
class FVGReport:
    symbol: str
    fvgs: list[FairValueGap]   # all gaps, newest first
    active_fvgs: list[FairValueGap]  # unfilled gaps
    bullish_count: int
    bearish_count: int
    nearest_bullish_gap: FairValueGap | None  # closest above/below current price
    nearest_bearish_gap: FairValueGap | None
    current_price: float
    bars_scanned: int
    verdict: str


def detect(bars: list, *, symbol: str = "", min_size_pct: float = 0.1) -> FVGReport | None:
    """Detect Fair Value Gaps in bars.

    bars: list[Bar] with .high .low .close — at least 5 bars.
    symbol: ticker for display.
    min_size_pct: minimum gap size as % of price to report.
    Returns None if fewer than 5 bars.
    """
    if not bars or len(bars) < 5:
        return None

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    fvgs: list[FairValueGap] = []

    for i in range(1, len(bars) - 1):
        try:
            b1_high = float(bars[i - 1].high)
            b1_low = float(bars[i - 1].low)
            b3_high = float(bars[i + 1].high)
            b3_low = float(bars[i + 1].low)
        except Exception:
            continue

        # Bullish FVG: gap between candle 1 high and candle 3 low
        if b3_low > b1_high:
            upper = b3_low
            lower = b1_high
            size_pct = (upper - lower) / lower * 100 if lower > 0 else 0.0
            if size_pct >= min_size_pct:
                filled = current_price <= upper and current_price >= lower
                partial = current_price < upper and current_price > lower
                fvgs.append(FairValueGap(
                    kind="bullish",
                    bar_index=i,
                    upper=round(upper, 2),
                    lower=round(lower, 2),
                    midpoint=round((upper + lower) / 2, 2),
                    size_pct=round(size_pct, 3),
                    filled=current_price < lower,
                    partial=filled and not (current_price < lower),
                ))

        # Bearish FVG: gap between candle 3 high and candle 1 low
        elif b3_high < b1_low:
            upper = b1_low
            lower = b3_high
            size_pct = (upper - lower) / lower * 100 if lower > 0 else 0.0
            if size_pct >= min_size_pct:
                filled = current_price >= lower and current_price <= upper
                fvgs.append(FairValueGap(
                    kind="bearish",
                    bar_index=i,
                    upper=round(upper, 2),
                    lower=round(lower, 2),
                    midpoint=round((upper + lower) / 2, 2),
                    size_pct=round(size_pct, 3),
                    filled=current_price > upper,
                    partial=filled,
                ))

    # Sort newest first
    fvgs.sort(key=lambda g: g.bar_index, reverse=True)
    active = [g for g in fvgs if not g.filled]
    bullish = [g for g in fvgs if g.kind == "bullish"]
    bearish = [g for g in fvgs if g.kind == "bearish"]

    # Nearest gaps relative to current price
    bullish_below = [g for g in active if g.kind == "bullish" and g.upper < current_price]
    bearish_above = [g for g in active if g.kind == "bearish" and g.lower > current_price]

    nearest_bull = max(bullish_below, key=lambda g: g.upper) if bullish_below else None
    nearest_bear = min(bearish_above, key=lambda g: g.lower) if bearish_above else None

    if not fvgs:
        verdict = f"No Fair Value Gaps detected in {len(bars)} bars."
    else:
        active_str = f"{len(active)} active" if active else "all filled"
        verdict = (
            f"{len(fvgs)} FVGs found ({active_str}): "
            f"{len(bullish)} bullish, {len(bearish)} bearish. "
        )
        if nearest_bull:
            verdict += f"Nearest support gap: {nearest_bull.lower:.2f}–{nearest_bull.upper:.2f}. "
        if nearest_bear:
            verdict += f"Nearest resistance gap: {nearest_bear.lower:.2f}–{nearest_bear.upper:.2f}."

    return FVGReport(
        symbol=symbol,
        fvgs=fvgs,
        active_fvgs=active,
        bullish_count=len(bullish),
        bearish_count=len(bearish),
        nearest_bullish_gap=nearest_bull,
        nearest_bearish_gap=nearest_bear,
        current_price=round(current_price, 2),
        bars_scanned=len(bars),
        verdict=verdict,
    )
