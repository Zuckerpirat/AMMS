"""Moving Average Ribbon analysis.

Computes a ribbon of 6 EMAs (5, 8, 13, 21, 34, 55) and determines
whether they are ordered (trending) or tangled (ranging/choppy).

Interpretation:
  - Ribbon ordered ascending (5>8>13>21>34>55): strong uptrend
  - Ribbon ordered descending (5<8<13<21<34<55): strong downtrend
  - Ribbon tangled (mixed order): ranging/transitional market

Also computes:
  - Ribbon spread (gap between fastest and slowest EMA as %)
  - Momentum direction (is the fast EMA pulling away or converging?)
  - Price position relative to ribbon (above all, below all, inside)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_PERIODS = [5, 8, 13, 21, 34, 55]


@dataclass(frozen=True)
class MARibbonReport:
    symbol: str
    emas: dict[int, float]        # period -> EMA value
    periods: list[int]
    ordered: bool                 # True if all EMAs are in strict order
    direction: str                # "up" / "down" / "tangled"
    ribbon_spread_pct: float      # (max_ema - min_ema) / mid_ema * 100
    price_position: str           # "above_all" / "below_all" / "inside" / "at_ribbon"
    is_expanding: bool            # ribbon spread increasing (trend strengthening)
    current_price: float
    bars_used: int
    verdict: str


def _ema_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(closes[:period]) / period]
    for v in closes[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def analyze(bars: list, *, symbol: str = "", periods: list[int] | None = None) -> MARibbonReport | None:
    """Analyze MA ribbon from bars.

    bars: list[Bar] with .close — at least max(periods)+5 bars.
    symbol: ticker for display.
    periods: EMA periods (default [5, 8, 13, 21, 34, 55]).
    """
    if periods is None:
        periods = DEFAULT_PERIODS
    periods = sorted(periods)
    min_bars = max(periods) + 5

    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
        current_price = closes[-1]
    except Exception:
        return None

    emas: dict[int, float] = {}
    for p in periods:
        series = _ema_series(closes, p)
        if series:
            emas[p] = round(series[-1], 4)

    if len(emas) < 2:
        return None

    ema_values = [emas[p] for p in periods if p in emas]
    valid_periods = [p for p in periods if p in emas]

    # Check ordering
    ascending = all(ema_values[i] > ema_values[i + 1] for i in range(len(ema_values) - 1))
    descending = all(ema_values[i] < ema_values[i + 1] for i in range(len(ema_values) - 1))
    ordered = ascending or descending
    direction = "up" if ascending else ("down" if descending else "tangled")

    max_ema = max(ema_values)
    min_ema = min(ema_values)
    mid_ema = sum(ema_values) / len(ema_values)
    spread_pct = (max_ema - min_ema) / mid_ema * 100 if mid_ema > 0 else 0.0

    # Price position
    if current_price > max_ema:
        price_pos = "above_all"
    elif current_price < min_ema:
        price_pos = "below_all"
    elif all(abs(current_price - v) / v < 0.005 for v in ema_values):
        price_pos = "at_ribbon"
    else:
        price_pos = "inside"

    # Is ribbon expanding? Compare current spread to past spread
    is_expanding = False
    if len(closes) > min_bars + 5:
        older_emas = []
        for p in valid_periods:
            series = _ema_series(closes[:-5], p)
            if series:
                older_emas.append(series[-1])
        if len(older_emas) >= 2:
            older_spread = (max(older_emas) - min(older_emas)) / (sum(older_emas) / len(older_emas)) * 100
            is_expanding = spread_pct > older_spread

    dir_desc = {
        "up": f"Strong uptrend — ribbon ordered ascending ({valid_periods[0]}>...>{valid_periods[-1]})",
        "down": f"Strong downtrend — ribbon ordered descending",
        "tangled": "Ranging/choppy — ribbon tangled (mixed order)",
    }.get(direction, direction)

    verdict = (
        f"{dir_desc}. "
        f"Spread: {spread_pct:.2f}%. "
        f"Price is {price_pos.replace('_', ' ')} the ribbon. "
        f"Ribbon {'expanding' if is_expanding else 'contracting'} — "
        f"trend {'strengthening' if is_expanding else 'weakening or ranging'}."
    )

    return MARibbonReport(
        symbol=symbol,
        emas={p: emas[p] for p in valid_periods},
        periods=valid_periods,
        ordered=ordered,
        direction=direction,
        ribbon_spread_pct=round(spread_pct, 3),
        price_position=price_pos,
        is_expanding=is_expanding,
        current_price=round(current_price, 2),
        bars_used=len(bars),
        verdict=verdict,
    )
