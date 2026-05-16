"""Fibonacci Retracement Level Analyser.

Computes the classic Fibonacci retracement and extension levels from the
most recent significant swing high and swing low in bar history.

Standard retracement levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
Extension levels: 127.2%, 161.8%, 200%, 261.8%

For each level the analyser reports:
  - Price at that level
  - Distance from current price (%)
  - Whether price is currently near this level (within threshold)

The "nearest level" becomes the key support/resistance reference.
"""

from __future__ import annotations

from dataclasses import dataclass

_RETRACEMENT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
_EXTENSION_RATIOS = [1.272, 1.618, 2.0, 2.618]


@dataclass(frozen=True)
class FibLevel:
    ratio: float        # 0.0 to 2.618
    label: str          # "0%", "23.6%", etc.
    price: float
    pct_from_current: float  # negative = below current price
    is_near: bool            # within near_threshold_pct
    kind: str           # "retracement" or "extension"


@dataclass(frozen=True)
class FibRetracementReport:
    symbol: str
    swing_high: float
    swing_low: float
    swing_high_idx: int
    swing_low_idx: int
    trend_direction: str   # "up" (low before high) or "down" (high before low)
    current_price: float
    levels: list[FibLevel]
    nearest_level: FibLevel | None
    nearest_support: FibLevel | None   # closest level below current price
    nearest_resistance: FibLevel | None  # closest level above current price
    retracement_depth: float   # how far price has retraced from swing (0-100%)
    bars_used: int
    near_threshold_pct: float
    verdict: str


def _find_swing_high(closes: list[float], highs: list[float], window: int = 50) -> tuple[int, float]:
    """Return (idx, value) of the highest bar in the last `window` bars."""
    subset = highs[-window:]
    offset = max(0, len(highs) - window)
    idx_local = subset.index(max(subset))
    return offset + idx_local, subset[idx_local]


def _find_swing_low(closes: list[float], lows: list[float], window: int = 50) -> tuple[int, float]:
    """Return (idx, value) of the lowest bar in the last `window` bars."""
    subset = lows[-window:]
    offset = max(0, len(lows) - window)
    idx_local = subset.index(min(subset))
    return offset + idx_local, subset[idx_local]


def analyze(
    bars: list,
    *,
    symbol: str = "",
    swing_window: int = 50,
    near_threshold_pct: float = 1.5,
) -> FibRetracementReport | None:
    """Analyse Fibonacci retracement levels from bar history.

    bars: bar objects with .close, .high, .low attributes.
    swing_window: how many recent bars to search for swing high/low.
    near_threshold_pct: % range to call a level "near".
    """
    if not bars or len(bars) < max(swing_window, 10) + 5:
        return None

    try:
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    hi_idx, hi_val = _find_swing_high(closes, highs, swing_window)
    lo_idx, lo_val = _find_swing_low(closes, lows, swing_window)

    if hi_val <= lo_val:
        return None

    # Determine trend: if high comes after low → up-trend retracement
    # if low comes after high → down-trend retracement
    trend = "up" if lo_idx < hi_idx else "down"
    swing_range = hi_val - lo_val

    # Retracement levels (from high toward low in up-trend, from low toward high in down-trend)
    levels: list[FibLevel] = []

    for ratio in _RETRACEMENT_RATIOS:
        if trend == "up":
            price = hi_val - ratio * swing_range
        else:
            price = lo_val + ratio * swing_range
        label = f"{ratio * 100:.1f}%".rstrip("0").rstrip(".")
        pct_from = (price - current) / current * 100.0 if current > 0 else 0.0
        is_near = abs(pct_from) <= near_threshold_pct
        levels.append(FibLevel(
            ratio=ratio,
            label=label,
            price=round(price, 4),
            pct_from_current=round(pct_from, 3),
            is_near=is_near,
            kind="retracement",
        ))

    for ratio in _EXTENSION_RATIOS:
        if trend == "up":
            price = hi_val + (ratio - 1.0) * swing_range
        else:
            price = lo_val - (ratio - 1.0) * swing_range
        label = f"{ratio * 100:.1f}%".rstrip("0").rstrip(".")
        pct_from = (price - current) / current * 100.0 if current > 0 else 0.0
        is_near = abs(pct_from) <= near_threshold_pct
        levels.append(FibLevel(
            ratio=ratio,
            label=label,
            price=round(price, 4),
            pct_from_current=round(pct_from, 3),
            is_near=is_near,
            kind="extension",
        ))

    # Find nearest level
    nearest = min(levels, key=lambda l: abs(l.pct_from_current))

    # Support: closest level below current
    below = [l for l in levels if l.price < current]
    support = max(below, key=lambda l: l.price) if below else None

    # Resistance: closest level above current
    above = [l for l in levels if l.price > current]
    resistance = min(above, key=lambda l: l.price) if above else None

    # Retracement depth: how far from swing high (up-trend) or swing low (down-trend)
    if trend == "up":
        depth = (hi_val - current) / swing_range * 100.0 if swing_range > 0 else 0.0
    else:
        depth = (current - lo_val) / swing_range * 100.0 if swing_range > 0 else 0.0
    depth = max(0.0, min(100.0, depth))

    # Verdict
    parts = []
    parts.append(f"{trend}-trend swing {lo_val:.2f}–{hi_val:.2f}")
    if nearest.is_near:
        parts.append(f"at Fib {nearest.label} ({nearest.price:.2f})")
    else:
        parts.append(f"nearest Fib {nearest.label} ({nearest.pct_from_current:+.1f}%)")
    if support:
        parts.append(f"support {support.label} ({support.price:.2f})")
    if resistance:
        parts.append(f"resistance {resistance.label} ({resistance.price:.2f})")
    parts.append(f"retracement depth {depth:.0f}%")

    verdict = "Fibonacci levels: " + "; ".join(parts) + "."

    return FibRetracementReport(
        symbol=symbol,
        swing_high=round(hi_val, 4),
        swing_low=round(lo_val, 4),
        swing_high_idx=hi_idx,
        swing_low_idx=lo_idx,
        trend_direction=trend,
        current_price=round(current, 4),
        levels=levels,
        nearest_level=nearest,
        nearest_support=support,
        nearest_resistance=resistance,
        retracement_depth=round(depth, 2),
        bars_used=len(bars),
        near_threshold_pct=near_threshold_pct,
        verdict=verdict,
    )
