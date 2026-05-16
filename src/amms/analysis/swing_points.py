"""Swing High/Low Detector and Analyser.

Identifies significant swing highs and lows in price action using a
configurable look-left/look-right pivot detection algorithm.
Uses these to determine market structure (higher highs/lows = uptrend, etc.)

Key outputs:
  - List of confirmed swing points (high/low, bar index, price level)
  - Market structure: HH/HL (uptrend), LH/LL (downtrend), sideways
  - Nearest support and resistance levels
  - Trend based on swing point sequence
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwingPoint:
    bar_idx: int
    price: float
    kind: str       # "high" or "low"
    strength: int   # how many bars it was the highest/lowest on each side


@dataclass(frozen=True)
class SwingReport:
    symbol: str
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]

    recent_high: SwingPoint | None   # most recent swing high
    recent_low: SwingPoint | None    # most recent swing low
    prior_high: SwingPoint | None    # second-most recent swing high
    prior_low: SwingPoint | None     # second-most recent swing low

    structure: str          # "uptrend", "downtrend", "sideways", "unknown"
    hh: bool                # higher highs vs prior high
    hl: bool                # higher lows vs prior low
    lh: bool                # lower highs vs prior high
    ll: bool                # lower lows vs prior low

    nearest_support: float | None    # nearest swing low below price
    nearest_resistance: float | None # nearest swing high above price
    support_distance_pct: float | None
    resistance_distance_pct: float | None

    total_swings: int
    current_price: float
    bars_used: int
    verdict: str


def _find_swings(highs: list[float], lows: list[float], lookback: int = 5) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Find pivot highs and lows using look-left/look-right windows."""
    n = len(highs)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(lookback, n - lookback):
        # Check if highs[i] is the highest in the window
        window_h = highs[i - lookback: i + lookback + 1]
        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            swing_highs.append(SwingPoint(
                bar_idx=i,
                price=round(highs[i], 4),
                kind="high",
                strength=lookback,
            ))

        # Check if lows[i] is the lowest in the window
        window_l = lows[i - lookback: i + lookback + 1]
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            swing_lows.append(SwingPoint(
                bar_idx=i,
                price=round(lows[i], 4),
                kind="low",
                strength=lookback,
            ))

    return swing_highs, swing_lows


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 5,
) -> SwingReport | None:
    """Detect swing highs/lows and determine market structure.

    bars: bar objects with .high, .low, .close attributes.
    lookback: bars to look left and right for pivot confirmation.
    """
    min_bars = lookback * 2 + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    swing_highs, swing_lows = _find_swings(highs, lows, lookback)

    recent_high = swing_highs[-1] if swing_highs else None
    recent_low = swing_lows[-1] if swing_lows else None
    prior_high = swing_highs[-2] if len(swing_highs) >= 2 else None
    prior_low = swing_lows[-2] if len(swing_lows) >= 2 else None

    # Market structure
    hh = recent_high is not None and prior_high is not None and recent_high.price > prior_high.price
    hl = recent_low is not None and prior_low is not None and recent_low.price > prior_low.price
    lh = recent_high is not None and prior_high is not None and recent_high.price < prior_high.price
    ll = recent_low is not None and prior_low is not None and recent_low.price < prior_low.price

    if hh and hl:
        structure = "uptrend"
    elif lh and ll:
        structure = "downtrend"
    elif (hh or hl) and not (lh or ll):
        structure = "uptrend"
    elif (lh or ll) and not (hh or hl):
        structure = "downtrend"
    elif len(swing_highs) >= 2 or len(swing_lows) >= 2:
        structure = "sideways"
    else:
        structure = "unknown"

    # Nearest support/resistance
    supports = [sp.price for sp in swing_lows if sp.price < current]
    resistances = [sp.price for sp in swing_highs if sp.price > current]

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    support_dist = (current - nearest_support) / current * 100.0 if nearest_support else None
    resistance_dist = (nearest_resistance - current) / current * 100.0 if nearest_resistance else None

    # Verdict
    parts = [f"Market structure: {structure}"]
    if hh:
        parts.append("HH ✓")
    if hl:
        parts.append("HL ✓")
    if lh:
        parts.append("LH")
    if ll:
        parts.append("LL")
    if nearest_support:
        parts.append(f"support {nearest_support:.2f} ({support_dist:.1f}% below)")
    if nearest_resistance:
        parts.append(f"resistance {nearest_resistance:.2f} ({resistance_dist:.1f}% above)")

    verdict = "Swing structure: " + ", ".join(parts) + f". {len(swing_highs)}H/{len(swing_lows)}L pivots found."

    return SwingReport(
        symbol=symbol,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        recent_high=recent_high,
        recent_low=recent_low,
        prior_high=prior_high,
        prior_low=prior_low,
        structure=structure,
        hh=hh,
        hl=hl,
        lh=lh,
        ll=ll,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        support_distance_pct=round(support_dist, 2) if support_dist is not None else None,
        resistance_distance_pct=round(resistance_dist, 2) if resistance_dist is not None else None,
        total_swings=len(swing_highs) + len(swing_lows),
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
