"""Price Target Consensus from multiple technical methods.

Aggregates upside and downside price targets from:
  1. Fibonacci extensions (1.272×, 1.618×, 2.000× from recent swing)
  2. ATR-based targets (price ± N × ATR)
  3. Classic pivot level projections (R2, R3 for upside; S2, S3 for downside)
  4. Measured move targets (prior swing amplitude projected from breakout)

Clusters nearby targets into a consensus zone (±0.5% of median).
Reports the strongest upside and downside consensus levels.

Interpretation:
  - Cluster of upside targets at the same price = strong resistance / take-profit zone
  - Cluster of downside targets = strong support / stop zone
  - Wide spread = no consensus (uncertain targets)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetLevel:
    price: float
    source: str          # e.g. "Fib 1.618" / "ATR×3" / "Pivot R2" / "Measured"
    direction: str       # "up" / "down"
    pct_from_current: float


@dataclass(frozen=True)
class ConsensusZone:
    center: float
    low: float
    high: float
    targets: list[TargetLevel]   # all targets in this cluster
    n_sources: int               # number of distinct methods
    direction: str               # "up" / "down"
    pct_from_current: float


@dataclass(frozen=True)
class PriceTargetConsensusReport:
    symbol: str
    current_price: float
    all_targets: list[TargetLevel]
    upside_targets: list[TargetLevel]
    downside_targets: list[TargetLevel]
    best_upside_zone: ConsensusZone | None
    best_downside_zone: ConsensusZone | None
    upside_consensus_pct: float    # how tight the upside cluster is (0 = spread, 100 = perfect)
    downside_consensus_pct: float
    bars_used: int
    verdict: str


def _atr(bars: list, period: int = 14) -> float:
    """Simple ATR estimate."""
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h = float(bars[-i].high)
        l = float(bars[-i].low)
        pc = float(bars[-i - 1].close)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else 0.0


def _swing_points(bars: list, window: int = 5) -> tuple[float, float, float, float]:
    """Return (recent_swing_high, recent_swing_low, swing_start_high, swing_start_low)."""
    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    # Recent swing (last 1/3 of bars)
    n = len(bars)
    recent_slice = max(window, n // 3)
    recent_high = max(highs[-recent_slice:])
    recent_low = min(lows[-recent_slice:])
    # Older swing (first 2/3)
    older_high = max(highs[:-recent_slice]) if len(highs) > recent_slice else recent_high
    older_low = min(lows[:-recent_slice]) if len(lows) > recent_slice else recent_low
    return recent_high, recent_low, older_high, older_low


def _cluster(targets: list[TargetLevel], cluster_pct: float = 0.5) -> list[ConsensusZone]:
    """Group targets within cluster_pct% of each other into consensus zones."""
    if not targets:
        return []

    sorted_t = sorted(targets, key=lambda t: t.price)
    zones: list[ConsensusZone] = []
    current_group = [sorted_t[0]]

    for t in sorted_t[1:]:
        ref = current_group[0].price
        if ref > 0 and abs(t.price - ref) / ref * 100 <= cluster_pct:
            current_group.append(t)
        else:
            zones.append(_make_zone(current_group))
            current_group = [t]

    if current_group:
        zones.append(_make_zone(current_group))

    return zones


def _make_zone(targets: list[TargetLevel]) -> ConsensusZone:
    prices = [t.price for t in targets]
    center = sum(prices) / len(prices)
    sources = {t.source.split(" ")[0] for t in targets}
    direction = targets[0].direction
    ref = targets[0].price
    pct = targets[0].pct_from_current
    return ConsensusZone(
        center=round(center, 4),
        low=round(min(prices), 4),
        high=round(max(prices), 4),
        targets=targets,
        n_sources=len(sources),
        direction=direction,
        pct_from_current=round(pct, 2),
    )


def analyze(
    bars: list,
    *,
    symbol: str = "",
    atr_multiples: tuple[float, ...] = (2.0, 3.0),
    fib_levels: tuple[float, ...] = (1.272, 1.618, 2.0),
) -> PriceTargetConsensusReport | None:
    """Derive price target consensus from multiple technical methods.

    bars: list[Bar] with .high .low .close — at least 20 bars.
    symbol: ticker for display.
    """
    if not bars or len(bars) < 20:
        return None

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    if current_price <= 0:
        return None

    try:
        atr = _atr(bars, period=14)
        recent_high, recent_low, older_high, older_low = _swing_points(bars)
    except Exception:
        return None

    targets: list[TargetLevel] = []

    def _add(price: float, source: str) -> None:
        if price <= 0:
            return
        pct = (price - current_price) / current_price * 100.0
        direction = "up" if pct > 0 else "down"
        targets.append(TargetLevel(
            price=round(price, 4),
            source=source,
            direction=direction,
            pct_from_current=round(pct, 2),
        ))

    # 1. Fibonacci extensions from recent swing low to recent swing high (upside)
    swing_range = recent_high - recent_low
    if swing_range > 0:
        for fib in fib_levels:
            _add(recent_low + swing_range * fib, f"Fib {fib:.3f}U")
        # Fibonacci retracements downside from recent high
        for fib in (0.382, 0.500, 0.618):
            _add(recent_high - swing_range * fib, f"Fib {fib:.3f}D")

    # 2. ATR-based targets
    if atr > 0:
        for mult in atr_multiples:
            _add(current_price + atr * mult, f"ATR×{mult:.1f}U")
            _add(current_price - atr * mult, f"ATR×{mult:.1f}D")

    # 3. Pivot projections (simplified classic pivots from recent range)
    pp = (recent_high + recent_low + float(bars[-1].close)) / 3.0
    pivot_range = recent_high - recent_low
    if pivot_range > 0:
        _add(pp + pivot_range, "Pivot R2")
        _add(pp + pivot_range * 1.5, "Pivot R3")
        _add(pp - pivot_range, "Pivot S2")
        _add(pp - pivot_range * 1.5, "Pivot S3")

    # 4. Measured move target (project older swing amplitude from current)
    old_range = older_high - older_low
    if old_range > 0:
        _add(current_price + old_range, "Measured U")
        _add(current_price - old_range, "Measured D")

    if not targets:
        return None

    upside = [t for t in targets if t.pct_from_current > 0.5]
    downside = [t for t in targets if t.pct_from_current < -0.5]

    up_zones = sorted(_cluster(upside), key=lambda z: -z.n_sources)
    down_zones = sorted(_cluster(downside), key=lambda z: -z.n_sources)

    best_up = up_zones[0] if up_zones else None
    best_down = down_zones[0] if down_zones else None

    def _consensus_pct(zones: list[ConsensusZone]) -> float:
        if not zones:
            return 0.0
        # Ratio: largest cluster / all targets in direction
        biggest = max(len(z.targets) for z in zones)
        total = sum(len(z.targets) for z in zones)
        return biggest / total * 100.0 if total > 0 else 0.0

    up_consensus = _consensus_pct(up_zones)
    down_consensus = _consensus_pct(down_zones)

    # Verdict
    up_desc = (
        f"Upside consensus at {best_up.center:.2f} (+{best_up.pct_from_current:.1f}%, "
        f"{best_up.n_sources} methods)"
        if best_up else "No clear upside consensus"
    )
    down_desc = (
        f"Downside consensus at {best_down.center:.2f} ({best_down.pct_from_current:.1f}%, "
        f"{best_down.n_sources} methods)"
        if best_down else "No clear downside consensus"
    )
    verdict = f"{up_desc}. {down_desc}."

    return PriceTargetConsensusReport(
        symbol=symbol,
        current_price=round(current_price, 2),
        all_targets=targets,
        upside_targets=upside,
        downside_targets=downside,
        best_upside_zone=best_up,
        best_downside_zone=best_down,
        upside_consensus_pct=round(up_consensus, 1),
        downside_consensus_pct=round(down_consensus, 1),
        bars_used=len(bars),
        verdict=verdict,
    )
