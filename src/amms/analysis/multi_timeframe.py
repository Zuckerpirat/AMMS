"""Multi-timeframe trend analysis.

Evaluates trend direction and strength across multiple timeframes from a
single bar series by computing indicators over different lookback windows.

Three timeframe tiers (derived from the same bars):
  - Short:  last N/4  bars — recent momentum
  - Medium: last N/2  bars — intermediate trend
  - Long:   all N     bars — dominant trend

For each tier:
  - Price vs EMA (above = bullish, below = bearish)
  - Linear regression slope (normalized)
  - ADX-proxy: ratio of directional movement to range

Alignment score: if all 3 tiers agree → strong confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeTrend:
    label: str            # "Short", "Medium", "Long"
    n_bars: int
    direction: str        # "up" / "down" / "flat"
    strength: float       # 0-100
    ema: float
    price_vs_ema: str     # "above" / "below" / "at"
    slope_pct: float      # regression slope as % of price per bar


@dataclass(frozen=True)
class MultiTimeframeReport:
    symbol: str
    tiers: list[TimeframeTrend]
    aligned: bool                 # all tiers same direction
    alignment_score: float        # 0-100 (100 = all aligned)
    dominant_direction: str       # direction agreed by majority of tiers
    current_price: float
    bars_available: int
    verdict: str


def _ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    result = sum(closes[:period]) / period
    for v in closes[period:]:
        result = v * k + result * (1 - k)
    return result


def _slope(closes: list[float]) -> float:
    """Linear regression slope as % of last price per bar."""
    n = len(closes)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(closes) / n
    ss_xy = sum((xs[i] - mx) * (closes[i] - my) for i in range(n))
    ss_xx = sum((x - mx) ** 2 for x in xs)
    if ss_xx == 0 or closes[-1] == 0:
        return 0.0
    return (ss_xy / ss_xx) / closes[-1] * 100


def _adx_proxy(bars: list, period: int = 14) -> float:
    """Simplified directional movement strength (0-100)."""
    if len(bars) < period + 1:
        return 50.0
    recent = bars[-period - 1:]
    dm_plus = []
    dm_minus = []
    for i in range(1, len(recent)):
        up_move = float(recent[i].high) - float(recent[i - 1].high)
        down_move = float(recent[i - 1].low) - float(recent[i].low)
        dm_plus.append(max(up_move, 0) if up_move > down_move else 0)
        dm_minus.append(max(down_move, 0) if down_move > up_move else 0)
    total = sum(dm_plus) + sum(dm_minus)
    if total == 0:
        return 0.0
    directional = abs(sum(dm_plus) - sum(dm_minus)) / total * 100
    return directional


def _analyze_tier(bars: list, label: str) -> TimeframeTrend | None:
    if len(bars) < 5:
        return None
    closes = [float(b.close) for b in bars]
    n = len(closes)
    current = closes[-1]

    period = min(max(5, n // 3), n - 1)
    ema_val = _ema(closes, period)
    if ema_val is None:
        return None

    if current > ema_val * 1.002:
        pve = "above"
    elif current < ema_val * 0.998:
        pve = "below"
    else:
        pve = "at"

    slope = _slope(closes)
    strength = _adx_proxy(bars)

    if slope > 0.05 or (slope > 0 and pve == "above"):
        direction = "up"
    elif slope < -0.05 or (slope < 0 and pve == "below"):
        direction = "down"
    else:
        direction = "flat"

    return TimeframeTrend(
        label=label,
        n_bars=n,
        direction=direction,
        strength=round(strength, 1),
        ema=round(ema_val, 2),
        price_vs_ema=pve,
        slope_pct=round(slope, 4),
    )


def analyze(bars: list, *, symbol: str = "") -> MultiTimeframeReport | None:
    """Analyze trend across 3 derived timeframes.

    bars: list[Bar] with .high .low .close — at least 20 bars.
    symbol: ticker for display.
    """
    if not bars or len(bars) < 20:
        return None

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    n = len(bars)
    short_n = max(5, n // 4)
    medium_n = max(10, n // 2)

    tiers_raw = [
        _analyze_tier(bars[-short_n:], "Short"),
        _analyze_tier(bars[-medium_n:], "Medium"),
        _analyze_tier(bars, "Long"),
    ]
    tiers = [t for t in tiers_raw if t is not None]

    if not tiers:
        return None

    directions = [t.direction for t in tiers]
    up = directions.count("up")
    down = directions.count("down")
    flat = directions.count("flat")

    if up > down and up > flat:
        dominant = "up"
    elif down > up and down > flat:
        dominant = "down"
    else:
        dominant = "flat"

    aligned = len(set(d for d in directions if d != "flat")) <= 1
    alignment_score = max(up, down, flat) / len(tiers) * 100

    dir_str = {"up": "bullish ↑", "down": "bearish ↓", "flat": "sideways ↔"}.get(dominant, dominant)
    if aligned and dominant != "flat":
        verdict = f"All timeframes aligned {dir_str} — high confidence directional bias."
    elif dominant == "flat":
        verdict = "No clear directional bias — mixed timeframe signals."
    else:
        verdict = f"Dominant trend: {dir_str} ({max(up, down, flat)}/{len(tiers)} tiers). Conflicting signals — proceed with caution."

    return MultiTimeframeReport(
        symbol=symbol,
        tiers=tiers,
        aligned=aligned,
        alignment_score=round(alignment_score, 1),
        dominant_direction=dominant,
        current_price=round(current_price, 2),
        bars_available=n,
        verdict=verdict,
    )
