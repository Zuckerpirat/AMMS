"""Support and resistance level detection.

Identifies horizontal price levels where price has repeatedly reversed,
using a density-based clustering approach on pivot highs and lows.

Algorithm:
  1. Detect pivot highs and lows (local extremes within a window)
  2. Collect all pivot prices into one pool
  3. Cluster nearby pivots (within tolerance_pct of each other)
  4. Score each cluster by: number of touches + recency + proximity to current price
  5. Return the top N levels, classified as support or resistance

A level is classified as:
  - "support": cluster price is below current price
  - "resistance": cluster price is above current price
  - "at_price": cluster price is within tolerance of current price
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SRLevel:
    price: float
    kind: str            # "support"|"resistance"|"at_price"
    touches: int         # how many pivot points were in this cluster
    strength: float      # normalized 0-100
    distance_pct: float  # % distance from current price (signed)


@dataclass(frozen=True)
class SRResult:
    symbol: str
    current_price: float
    levels: list[SRLevel]       # sorted by price
    nearest_support: SRLevel | None
    nearest_resistance: SRLevel | None
    support_distance_pct: float | None
    resistance_distance_pct: float | None
    bars_used: int


def detect(
    bars: list,
    *,
    pivot_window: int = 3,
    lookback: int = 60,
    tolerance_pct: float = 0.5,
    top_n: int = 6,
) -> SRResult | None:
    """Detect support and resistance levels from bar data.

    bars: list[Bar]
    pivot_window: bars on each side to qualify as pivot (default 3)
    lookback: number of recent bars to analyze (default 60)
    tolerance_pct: cluster proximity threshold (default 0.5%)
    top_n: maximum number of levels to return (default 6)
    """
    if len(bars) < pivot_window * 2 + 3:
        return None

    symbol = bars[0].symbol
    window = bars[-lookback:] if len(bars) > lookback else bars
    n = len(window)
    current_price = window[-1].close

    # Collect pivot highs and lows
    pivot_prices: list[tuple[float, int]] = []  # (price, bar_index)
    for i in range(pivot_window, n - pivot_window):
        highs = [window[j].high for j in range(i - pivot_window, i + pivot_window + 1)]
        lows = [window[j].low for j in range(i - pivot_window, i + pivot_window + 1)]
        if window[i].high == max(highs):
            pivot_prices.append((window[i].high, i))
        if window[i].low == min(lows):
            pivot_prices.append((window[i].low, i))

    if not pivot_prices:
        return SRResult(
            symbol=symbol,
            current_price=round(current_price, 4),
            levels=[],
            nearest_support=None,
            nearest_resistance=None,
            support_distance_pct=None,
            resistance_distance_pct=None,
            bars_used=n,
        )

    # Cluster nearby pivots
    pivot_prices_sorted = sorted(pivot_prices, key=lambda x: x[0])
    clusters: list[list[tuple[float, int]]] = []
    current_cluster: list[tuple[float, int]] = [pivot_prices_sorted[0]]

    for price, idx in pivot_prices_sorted[1:]:
        cluster_center = sum(p for p, _ in current_cluster) / len(current_cluster)
        if abs(price - cluster_center) / cluster_center * 100 <= tolerance_pct:
            current_cluster.append((price, idx))
        else:
            clusters.append(current_cluster)
            current_cluster = [(price, idx)]
    clusters.append(current_cluster)

    # Score each cluster
    sr_levels: list[SRLevel] = []
    for cluster in clusters:
        avg_price = sum(p for p, _ in cluster) / len(cluster)
        touches = len(cluster)
        # Recency bonus: most recent touch gets higher weight
        most_recent_idx = max(idx for _, idx in cluster)
        recency_score = most_recent_idx / n  # 0..1
        raw_strength = touches * 10 + recency_score * 20

        distance_pct = (avg_price - current_price) / current_price * 100
        if abs(distance_pct) <= tolerance_pct:
            kind = "at_price"
        elif avg_price < current_price:
            kind = "support"
        else:
            kind = "resistance"

        sr_levels.append(SRLevel(
            price=round(avg_price, 4),
            kind=kind,
            touches=touches,
            strength=round(raw_strength, 1),
            distance_pct=round(distance_pct, 2),
        ))

    # Normalize strength to 0-100
    if sr_levels:
        max_strength = max(l.strength for l in sr_levels)
        if max_strength > 0:
            sr_levels = [
                SRLevel(
                    price=l.price,
                    kind=l.kind,
                    touches=l.touches,
                    strength=round(l.strength / max_strength * 100, 1),
                    distance_pct=l.distance_pct,
                )
                for l in sr_levels
            ]

    # Sort by strength descending, take top_n, then re-sort by price
    sr_levels.sort(key=lambda l: -l.strength)
    sr_levels = sr_levels[:top_n]
    sr_levels.sort(key=lambda l: l.price)

    supports = [l for l in sr_levels if l.kind == "support"]
    resistances = [l for l in sr_levels if l.kind == "resistance"]

    nearest_support = supports[-1] if supports else None  # closest = highest support
    nearest_resistance = resistances[0] if resistances else None  # closest = lowest resistance

    return SRResult(
        symbol=symbol,
        current_price=round(current_price, 4),
        levels=sr_levels,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        support_distance_pct=nearest_support.distance_pct if nearest_support else None,
        resistance_distance_pct=nearest_resistance.distance_pct if nearest_resistance else None,
        bars_used=n,
    )
