"""Market structure analysis.

Identifies market structure by detecting swing highs and swing lows,
then classifying the pattern as:
  - "uptrend":   higher highs (HH) and higher lows (HL)
  - "downtrend": lower highs (LH) and lower lows (LL)
  - "ranging":   mixed or no clear pattern
  - "broken":    structure recently broke (e.g., HH after downtrend)

Also computes:
  - Last N swing points
  - Structure break detection (CHoCH — change of character)
  - Distance from last swing high/low

Reads: list of bars with .high .low .close
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwingPoint:
    bar_index: int
    price: float
    kind: str    # "high" / "low"


@dataclass(frozen=True)
class MarketStructureReport:
    symbol: str
    structure: str              # "uptrend" / "downtrend" / "ranging" / "unclear"
    choch_detected: bool        # change of character detected
    choch_direction: str        # "bullish" / "bearish" / "none"
    swing_highs: list[float]    # last 3 swing highs
    swing_lows: list[float]     # last 3 swing lows
    last_swing_high: float | None
    last_swing_low: float | None
    current_price: float
    pct_from_last_high: float   # how far below last swing high
    pct_from_last_low: float    # how far above last swing low
    bars_used: int
    verdict: str


def _find_swings(bars: list, *, window: int = 3) -> list[SwingPoint]:
    """Find swing highs and lows using a simple peak/trough detector."""
    n = len(bars)
    swings: list[SwingPoint] = []
    for i in range(window, n - window):
        # Swing high: highest of surrounding window
        is_high = all(
            float(bars[i].high) >= float(bars[j].high)
            for j in range(i - window, i + window + 1) if j != i
        )
        # Swing low: lowest of surrounding window
        is_low = all(
            float(bars[i].low) <= float(bars[j].low)
            for j in range(i - window, i + window + 1) if j != i
        )
        if is_high:
            swings.append(SwingPoint(i, float(bars[i].high), "high"))
        elif is_low:
            swings.append(SwingPoint(i, float(bars[i].low), "low"))
    return swings


def _classify_structure(highs: list[float], lows: list[float]) -> str:
    if len(highs) < 2 or len(lows) < 2:
        return "unclear"
    hh = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    hl = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    lh = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    ll = all(lows[i] < lows[i - 1] for i in range(1, len(lows)))
    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    if (hh and ll) or (lh and hl):
        return "ranging"
    return "unclear"


def analyze(bars: list, *, symbol: str = "", swing_window: int = 3) -> MarketStructureReport | None:
    """Analyze market structure from bars.

    bars: list[Bar] with .high .low .close — at least 20 bars.
    symbol: ticker for display.
    swing_window: bars on each side to qualify a swing point.
    """
    min_bars = swing_window * 2 + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    swings = _find_swings(bars, window=swing_window)
    if len(swings) < 4:
        return None

    swing_highs = [s.price for s in swings if s.kind == "high"]
    swing_lows = [s.price for s in swings if s.kind == "low"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        structure = "unclear"
    else:
        structure = _classify_structure(swing_highs[-3:], swing_lows[-3:])

    # CHoCH (Change of Character) detection
    # Bullish CHoCH: was downtrend, now broke above last swing high
    # Bearish CHoCH: was uptrend, now broke below last swing low
    choch = False
    choch_dir = "none"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_high = swing_highs[-1]
        last_low = swing_lows[-1]
        if structure == "downtrend" and current_price > last_high:
            choch = True
            choch_dir = "bullish"
        elif structure == "uptrend" and current_price < last_low:
            choch = True
            choch_dir = "bearish"

    last_high = swing_highs[-1] if swing_highs else None
    last_low = swing_lows[-1] if swing_lows else None

    pct_from_high = (current_price / last_high - 1) * 100 if last_high and last_high > 0 else 0.0
    pct_from_low = (current_price / last_low - 1) * 100 if last_low and last_low > 0 else 0.0

    if choch:
        alert = f"⚠️ CHoCH ({choch_dir}) detected — potential trend change. "
    else:
        alert = ""

    struct_desc = {
        "uptrend": "Uptrend: HH + HL pattern intact",
        "downtrend": "Downtrend: LH + LL pattern intact",
        "ranging": "Ranging: mixed HH/LL pattern",
        "unclear": "Structure unclear — insufficient swing points",
    }.get(structure, structure)

    verdict = (
        f"{alert}{struct_desc}. "
        f"Last swing high: {last_high:.2f} ({pct_from_high:+.1f}% from current). "
        f"Last swing low: {last_low:.2f} ({pct_from_low:+.1f}% from current)."
        if last_high and last_low else f"{alert}{struct_desc}."
    )

    return MarketStructureReport(
        symbol=symbol,
        structure=structure,
        choch_detected=choch,
        choch_direction=choch_dir,
        swing_highs=swing_highs[-3:],
        swing_lows=swing_lows[-3:],
        last_swing_high=round(last_high, 2) if last_high else None,
        last_swing_low=round(last_low, 2) if last_low else None,
        current_price=round(current_price, 2),
        pct_from_last_high=round(pct_from_high, 2),
        pct_from_last_low=round(pct_from_low, 2),
        bars_used=len(bars),
        verdict=verdict,
    )
