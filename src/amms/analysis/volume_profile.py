"""Volume Profile (Volume at Price) analysis.

Distributes traded volume across price buckets to identify:
  - Point of Control (POC): price level with highest volume
  - Value Area (VA): 70% of total volume around POC
  - Value Area High (VAH) and Value Area Low (VAL)
  - High Volume Nodes (HVN): price levels with above-average volume
  - Low Volume Nodes (LVN): price gaps (potential fast-move zones)

Market profile interpretation:
  - Price above VAH → extended, potential mean reversion
  - Price below VAL → extended, potential mean reversion
  - Price at POC → fair value, balanced market
  - LVN below price → support gap (fast move if broken)
  - LVN above price → resistance gap (fast move if broken up)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VolumeNode:
    price: float
    volume: float
    volume_pct: float   # % of total volume
    is_hvn: bool        # High Volume Node
    is_lvn: bool        # Low Volume Node


@dataclass(frozen=True)
class VolumeProfile:
    symbol: str
    poc: float                    # Point of Control price
    vah: float                    # Value Area High
    val: float                    # Value Area Low
    current_price: float
    price_vs_va: str              # "above_va" / "in_va" / "below_va"
    price_vs_poc: str             # "above_poc" / "at_poc" / "below_poc"
    nodes: list[VolumeNode]       # all price buckets, sorted by price desc
    hvn_count: int
    lvn_count: int
    nearest_hvn_above: float | None
    nearest_hvn_below: float | None
    nearest_lvn_above: float | None
    nearest_lvn_below: float | None
    total_volume: float
    n_buckets: int
    bars_used: int
    verdict: str


def compute(
    bars: list,
    *,
    symbol: str = "",
    n_buckets: int = 24,
    value_area_pct: float = 0.70,
    hvn_threshold: float = 1.3,
    lvn_threshold: float = 0.4,
) -> VolumeProfile | None:
    """Build a volume profile from OHLCV bars.

    bars: list of bars with .high, .low, .close, .volume
    symbol: symbol name for display
    n_buckets: number of price buckets to use
    value_area_pct: fraction of volume defining the value area (default 70%)
    hvn_threshold: bucket volume / avg_volume ratio to flag as HVN
    lvn_threshold: bucket volume / avg_volume ratio to flag as LVN

    Returns None if fewer than 10 bars or no volume.
    """
    if not bars or len(bars) < 10:
        return None

    # Filter bars with valid volume
    valid = [b for b in bars if hasattr(b, "volume") and float(b.volume) > 0]
    if len(valid) < 10:
        return None

    # Price range
    price_high = max(float(b.high) for b in valid)
    price_low = min(float(b.low) for b in valid)
    if price_high <= price_low:
        return None

    bucket_size = (price_high - price_low) / n_buckets
    if bucket_size <= 0:
        return None

    # Distribute volume across price buckets using typical price (HLC/3)
    bucket_volume = [0.0] * n_buckets
    for b in valid:
        typical = (float(b.high) + float(b.low) + float(b.close)) / 3
        idx = int((typical - price_low) / bucket_size)
        idx = min(idx, n_buckets - 1)
        bucket_volume[idx] += float(b.volume)

    total_volume = sum(bucket_volume)
    if total_volume <= 0:
        return None

    avg_volume = total_volume / n_buckets

    # Point of Control: highest volume bucket
    poc_idx = bucket_volume.index(max(bucket_volume))
    poc = price_low + (poc_idx + 0.5) * bucket_size

    # Value Area: expand from POC until 70% of volume captured
    va_volume_target = total_volume * value_area_pct
    va_indices = {poc_idx}
    va_vol = bucket_volume[poc_idx]
    lo_ptr = poc_idx
    hi_ptr = poc_idx

    while va_vol < va_volume_target:
        lo_next = lo_ptr - 1
        hi_next = hi_ptr + 1
        add_lo = bucket_volume[lo_next] if lo_next >= 0 else -1
        add_hi = bucket_volume[hi_next] if hi_next < n_buckets else -1

        if add_lo <= 0 and add_hi <= 0:
            break
        if add_lo >= add_hi:
            lo_ptr = lo_next
            va_indices.add(lo_ptr)
            va_vol += add_lo
        else:
            hi_ptr = hi_next
            va_indices.add(hi_ptr)
            va_vol += add_hi

    val = price_low + lo_ptr * bucket_size
    vah = price_low + (hi_ptr + 1) * bucket_size

    # Build nodes
    nodes: list[VolumeNode] = []
    for i in range(n_buckets - 1, -1, -1):
        price = price_low + (i + 0.5) * bucket_size
        vol = bucket_volume[i]
        vol_pct = vol / total_volume * 100
        ratio = vol / avg_volume if avg_volume > 0 else 0
        is_hvn = ratio >= hvn_threshold
        is_lvn = ratio <= lvn_threshold and vol > 0
        nodes.append(VolumeNode(
            price=round(price, 2),
            volume=round(vol, 0),
            volume_pct=round(vol_pct, 2),
            is_hvn=is_hvn,
            is_lvn=is_lvn,
        ))

    # Current price
    current_price = float(valid[-1].close)

    # Position relative to value area
    if current_price > vah:
        price_vs_va = "above_va"
    elif current_price < val:
        price_vs_va = "below_va"
    else:
        price_vs_va = "in_va"

    poc_tol = bucket_size * 0.5
    if abs(current_price - poc) <= poc_tol:
        price_vs_poc = "at_poc"
    elif current_price > poc:
        price_vs_poc = "above_poc"
    else:
        price_vs_poc = "below_poc"

    # Nearest HVN and LVN above/below current price
    hvn_above = None
    hvn_below = None
    lvn_above = None
    lvn_below = None

    for node in nodes:  # sorted high to low
        if node.is_hvn:
            if node.price > current_price and hvn_above is None:
                hvn_above = node.price
            if node.price < current_price and hvn_below is None:
                hvn_below = node.price
        if node.is_lvn:
            if node.price > current_price and lvn_above is None:
                lvn_above = node.price
            if node.price < current_price and lvn_below is None:
                lvn_below = node.price

    hvn_count = sum(1 for n in nodes if n.is_hvn)
    lvn_count = sum(1 for n in nodes if n.is_lvn)

    # Verdict
    if price_vs_va == "above_va":
        verdict = "Price above Value Area — extended, watch for mean reversion to VAH"
    elif price_vs_va == "below_va":
        verdict = "Price below Value Area — extended, watch for recovery to VAL"
    elif price_vs_poc == "at_poc":
        verdict = "Price at Point of Control — balanced, market at fair value"
    else:
        verdict = "Price inside Value Area — balanced market conditions"

    return VolumeProfile(
        symbol=symbol,
        poc=round(poc, 2),
        vah=round(vah, 2),
        val=round(val, 2),
        current_price=round(current_price, 2),
        price_vs_va=price_vs_va,
        price_vs_poc=price_vs_poc,
        nodes=nodes,
        hvn_count=hvn_count,
        lvn_count=lvn_count,
        nearest_hvn_above=round(hvn_above, 2) if hvn_above else None,
        nearest_hvn_below=round(hvn_below, 2) if hvn_below else None,
        nearest_lvn_above=round(lvn_above, 2) if lvn_above else None,
        nearest_lvn_below=round(lvn_below, 2) if lvn_below else None,
        total_volume=round(total_volume, 0),
        n_buckets=n_buckets,
        bars_used=len(valid),
        verdict=verdict,
    )
