"""Volume Spread Analysis (VSA).

VSA examines the relationship between bar spread (range), closing position
within the bar, and volume to detect institutional accumulation, distribution,
and strength/weakness signals.

Key VSA bar types:
  - Up thrust:     Wide spread up, closes near low, high volume — bearish
  - No demand:     Narrow spread up, closes near low, low volume — bearish
  - Stopping vol:  Wide spread down, closes near high, high volume — bullish
  - No supply:     Narrow spread down, closes near high, low volume — bullish
  - Effort up:     Wide spread up, closes near high, high volume — bullish
  - Effort down:   Wide spread down, closes near low, high volume — bearish
  - Weakness:      Spread up but close in lower half, high volume
  - Strength:      Spread down but close in upper half, high volume
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VSABar:
    bar_idx: int
    price: float
    spread: float        # high - low
    close_position: float  # 0=closed at low, 1=closed at high
    volume: float
    relative_spread: float   # spread / avg_spread
    relative_volume: float   # volume / avg_volume
    label: str           # VSA bar type
    bias: str            # "bullish", "bearish", "neutral"
    strength: str        # "strong", "moderate", "weak"


@dataclass(frozen=True)
class VSAReport:
    symbol: str

    current_bar: VSABar
    recent_bars: list[VSABar]    # last N bars with labels

    bullish_count: int
    bearish_count: int
    dominant_bias: str      # "bullish", "bearish", "neutral"
    bias_score: float       # -1 to +1

    avg_spread: float
    avg_volume: float
    current_spread: float
    current_volume: float

    supply_detected: bool        # signs of distribution (supply)
    demand_detected: bool        # signs of accumulation (demand)

    current_price: float
    bars_used: int
    verdict: str


def _close_pos(open_: float, high: float, low: float, close: float) -> float:
    """Position of close within bar range: 0=at low, 1=at high."""
    rng = high - low
    if rng < 1e-9:
        return 0.5
    return (close - low) / rng


def _classify_bar(
    high: float, low: float, close: float, open_: float,
    volume: float, rel_spread: float, rel_vol: float
) -> tuple[str, str, str]:
    """Return (label, bias, strength)."""
    cp = _close_pos(open_, high, low, close)
    up_bar = close > open_
    wide = rel_spread > 1.3
    narrow = rel_spread < 0.7
    high_vol = rel_vol > 1.3
    low_vol = rel_vol < 0.7
    mid_vol = not high_vol and not low_vol

    # Up thrust: wide spread up, closes near LOW, high volume — distribution
    if up_bar and wide and cp < 0.35 and high_vol:
        return "Up Thrust", "bearish", "strong"

    # No demand: narrow spread up, closes near low, low volume
    if up_bar and narrow and cp < 0.5 and low_vol:
        return "No Demand", "bearish", "moderate"

    # Stopping volume: wide spread down, closes near HIGH, high volume — absorption
    if not up_bar and wide and cp > 0.65 and high_vol:
        return "Stopping Volume", "bullish", "strong"

    # No supply: narrow spread down, closes near high, low volume
    if not up_bar and narrow and cp > 0.5 and low_vol:
        return "No Supply", "bullish", "moderate"

    # Effort up: wide spread up, closes near high, high volume — bullish
    if up_bar and wide and cp > 0.65 and high_vol:
        return "Effort Up", "bullish", "strong"

    # Effort down: wide spread down, closes near low, high volume — bearish
    if not up_bar and wide and cp < 0.35 and high_vol:
        return "Effort Down", "bearish", "strong"

    # Weakness: up bar but closes in lower half, moderate/high volume
    if up_bar and cp < 0.4 and not low_vol:
        return "Hidden Weakness", "bearish", "moderate"

    # Strength: down bar but closes in upper half, moderate/high volume
    if not up_bar and cp > 0.6 and not low_vol:
        return "Hidden Strength", "bullish", "moderate"

    # Neutral bars
    if up_bar and cp > 0.5:
        return "Normal Up", "neutral", "weak"
    if not up_bar and cp < 0.5:
        return "Normal Down", "neutral", "weak"
    return "Neutral", "neutral", "weak"


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 20,
    recent_bars: int = 10,
) -> VSAReport | None:
    """Perform Volume Spread Analysis on bar history.

    bars: bar objects with .open, .high, .low, .close, .volume attributes.
    lookback: bars used to compute average spread and volume baselines.
    recent_bars: how many recent bars to label and return.
    """
    if not bars or len(bars) < lookback + 2:
        return None

    try:
        opens = [float(b.open) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)
    spreads = [highs[i] - lows[i] for i in range(n)]

    # Compute rolling averages for each bar using prior `lookback` bars
    def _rolling_avg(vals: list[float], idx: int, window: int) -> float:
        start = max(0, idx - window)
        subset = vals[start:idx]
        return sum(subset) / len(subset) if subset else vals[idx]

    # Label each bar in the recent window
    labeled: list[VSABar] = []
    scan_start = max(lookback, n - recent_bars)

    for i in range(scan_start, n):
        avg_sp = _rolling_avg(spreads, i, lookback)
        avg_vol = _rolling_avg(volumes, i, lookback)
        rel_sp = spreads[i] / avg_sp if avg_sp > 0 else 1.0
        rel_vol = volumes[i] / avg_vol if avg_vol > 0 else 1.0

        label, bias, strength = _classify_bar(
            highs[i], lows[i], closes[i], opens[i],
            volumes[i], rel_sp, rel_vol
        )
        cp = _close_pos(opens[i], highs[i], lows[i], closes[i])

        labeled.append(VSABar(
            bar_idx=i,
            price=round(closes[i], 4),
            spread=round(spreads[i], 4),
            close_position=round(cp, 3),
            volume=volumes[i],
            relative_spread=round(rel_sp, 3),
            relative_volume=round(rel_vol, 3),
            label=label,
            bias=bias,
            strength=strength,
        ))

    if not labeled:
        return None

    current_bar = labeled[-1]
    avg_sp_cur = _rolling_avg(spreads, n - 1, lookback)
    avg_vol_cur = _rolling_avg(volumes, n - 1, lookback)

    bull_count = sum(1 for b in labeled if b.bias == "bullish")
    bear_count = sum(1 for b in labeled if b.bias == "bearish")
    total_bias = bull_count + bear_count
    bias_score = (bull_count - bear_count) / total_bias if total_bias > 0 else 0.0

    if bias_score > 0.2:
        dominant = "bullish"
    elif bias_score < -0.2:
        dominant = "bearish"
    else:
        dominant = "neutral"

    # Supply/demand detection: recent strong signals
    supply = any(b.label in {"Up Thrust", "Effort Down", "Hidden Weakness"} and b.strength == "strong"
                 for b in labeled[-5:])
    demand = any(b.label in {"Stopping Volume", "Effort Up", "Hidden Strength"} and b.strength == "strong"
                 for b in labeled[-5:])

    # Verdict
    parts = [f"Current bar: {current_bar.label} ({current_bar.bias})"]
    if supply:
        parts.append("supply signals present")
    if demand:
        parts.append("demand signals present")
    parts.append(f"bias {dominant} ({bull_count}B/{bear_count}S in last {len(labeled)} bars)")
    verdict = "VSA: " + "; ".join(parts) + "."

    return VSAReport(
        symbol=symbol,
        current_bar=current_bar,
        recent_bars=labeled,
        bullish_count=bull_count,
        bearish_count=bear_count,
        dominant_bias=dominant,
        bias_score=round(bias_score, 3),
        avg_spread=round(avg_sp_cur, 4),
        avg_volume=round(avg_vol_cur, 0),
        current_spread=round(spreads[-1], 4),
        current_volume=volumes[-1],
        supply_detected=supply,
        demand_detected=demand,
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
