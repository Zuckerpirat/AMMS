"""Gap analysis — detects and classifies price gaps.

A gap occurs when the opening price of a bar is significantly different
from the previous bar's close. Gaps often indicate strong sentiment shifts.

Gap types:
  - Breakaway gap: occurs at the start of a new trend (after consolidation)
  - Continuation gap (runaway): occurs mid-trend, confirms trend strength
  - Exhaustion gap: occurs late in a trend, likely to be filled quickly
  - Common gap: small gap, fills quickly (normal market noise)

Since we can't algorithmically distinguish all types without context,
this module focuses on:
  - Gap detection (size, direction, fill status)
  - Gap fill probability scoring
  - Recent gap history (unfilled gaps still act as support/resistance)

A gap is "filled" when price returns to the gap zone within the lookback.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class Gap:
    bar_idx: int          # index in the bars list where gap occurred
    direction: str        # "up" | "down"
    gap_size: float       # absolute gap size (open - prev_close)
    gap_pct: float        # gap as % of prev_close
    prev_close: float
    open_price: float
    filled: bool          # whether gap was filled in subsequent bars


@dataclass(frozen=True)
class GapAnalysis:
    symbol: str
    current_price: float
    gaps: list[Gap]                # all detected gaps
    unfilled_gaps: list[Gap]       # gaps not yet filled (act as S/R)
    last_gap: Gap | None           # most recent gap
    nearest_gap_support: float | None   # nearest unfilled gap below price
    nearest_gap_resistance: float | None  # nearest unfilled gap above price
    bars_analyzed: int


def analyze_gaps(
    bars: list[Bar],
    *,
    min_gap_pct: float = 0.3,
    lookback: int = 60,
) -> GapAnalysis | None:
    """Detect and analyze price gaps in the bar series.

    min_gap_pct: minimum gap size as % of prev_close to be considered significant
    lookback: number of bars to scan for gaps
    Returns None if fewer than 3 bars.
    """
    if len(bars) < 3:
        return None

    symbol = bars[0].symbol
    window = bars[-lookback:] if len(bars) >= lookback else bars
    n = len(window)
    current_price = window[-1].close

    gaps: list[Gap] = []

    for i in range(1, n):
        prev_close = window[i - 1].close
        open_price = window[i].open

        if prev_close <= 0:
            continue

        diff = open_price - prev_close
        gap_pct = abs(diff) / prev_close * 100

        if gap_pct < min_gap_pct:
            continue

        direction = "up" if diff > 0 else "down"

        # Check if gap was filled in subsequent bars
        filled = False
        for j in range(i, n):
            if direction == "up":
                # Gap filled when price dips back to prev_close level
                if window[j].low <= prev_close:
                    filled = True
                    break
            else:
                # Gap filled when price rallies back to prev_close level
                if window[j].high >= prev_close:
                    filled = True
                    break

        gaps.append(Gap(
            bar_idx=i,
            direction=direction,
            gap_size=round(abs(diff), 4),
            gap_pct=round(gap_pct, 2),
            prev_close=round(prev_close, 4),
            open_price=round(open_price, 4),
            filled=filled,
        ))

    unfilled = [g for g in gaps if not g.filled]
    last_gap = gaps[-1] if gaps else None

    # Find nearest unfilled gaps above/below current price
    gap_support = None
    gap_resistance = None

    for g in unfilled:
        level = g.prev_close  # gap zone reference
        if level < current_price:
            if gap_support is None or level > gap_support:
                gap_support = level
        elif level > current_price:
            if gap_resistance is None or level < gap_resistance:
                gap_resistance = level

    return GapAnalysis(
        symbol=symbol,
        current_price=round(current_price, 4),
        gaps=gaps,
        unfilled_gaps=unfilled,
        last_gap=last_gap,
        nearest_gap_support=round(gap_support, 4) if gap_support is not None else None,
        nearest_gap_resistance=round(gap_resistance, 4) if gap_resistance is not None else None,
        bars_analyzed=n,
    )
