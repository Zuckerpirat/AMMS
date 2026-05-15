"""Volume-Weighted Average Price (VWAP) and Volume Profile.

VWAP = sum(typical_price * volume) / sum(volume)
Uses the typical price (high + low + close) / 3 per bar.

Also computes std dev bands (±1σ, ±2σ) around VWAP and a Volume Profile
with Point of Control (POC) and Value Area (VAH/VAL).
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class VWAPResult:
    symbol: str
    vwap: float
    current_price: float
    position: str          # "above_vwap" | "below_vwap" | "at_vwap"
    deviation_pct: float   # (price - vwap) / vwap * 100
    std1_upper: float      # vwap + 1σ
    std1_lower: float      # vwap - 1σ
    std2_upper: float      # vwap + 2σ
    std2_lower: float      # vwap - 2σ
    bars_used: int


@dataclass(frozen=True)
class VolumeProfileResult:
    symbol: str
    poc_price: float         # Point of Control — most-traded price level
    vah: float               # Value Area High (upper bound of 70% volume zone)
    val: float               # Value Area Low (lower bound of 70% volume zone)
    current_price: float
    poc_relation: str        # "above_poc" | "below_poc" | "at_poc"
    n_buckets: int
    bars_used: int


def vwap(bars: list[Bar], n: int | None = None) -> float | None:
    """Compute plain VWAP over the last n bars (or all bars if n is None).

    Returns None when there are no bars or total volume is zero.
    """
    window = bars[-n:] if n is not None and n > 0 else bars
    if not window:
        return None
    total_vol = sum(b.volume for b in window)
    if total_vol <= 0:
        return None
    total_pv = sum(((b.high + b.low + b.close) / 3) * b.volume for b in window)
    return total_pv / total_vol


def vwap_deviation_pct(price: float, bars: list[Bar], n: int | None = None) -> float | None:
    """Percentage deviation of price from VWAP.

    Positive = price above VWAP (momentum), negative = below (discount).
    Returns None when VWAP cannot be computed.
    """
    v = vwap(bars, n)
    if v is None or v <= 0:
        return None
    return (price - v) / v * 100


def vwap_full(bars: list[Bar], n: int | None = None) -> VWAPResult | None:
    """Compute VWAP with standard deviation bands.

    n: if given, use only the last n bars; otherwise all bars.
    Returns None if fewer than 5 bars or zero volume.
    """
    if len(bars) < 5:
        return None

    window = bars[-n:] if n is not None and len(bars) >= n else bars
    if len(window) < 2:
        return None

    symbol = window[0].symbol

    total_pv = 0.0
    total_vol = 0.0
    for b in window:
        tp = (b.high + b.low + b.close) / 3.0
        v = max(b.volume, 0.0)
        total_pv += tp * v
        total_vol += v

    if total_vol <= 0:
        return None

    vwap_val = total_pv / total_vol

    # Volume-weighted variance
    total_pv2 = sum(max(b.volume, 0.0) * ((b.high + b.low + b.close) / 3.0 - vwap_val) ** 2 for b in window)
    std = (total_pv2 / total_vol) ** 0.5

    price = window[-1].close
    dev_pct = (price - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0

    tol = std * 0.5
    if price > vwap_val + tol:
        position = "above_vwap"
    elif price < vwap_val - tol:
        position = "below_vwap"
    else:
        position = "at_vwap"

    return VWAPResult(
        symbol=symbol,
        vwap=round(vwap_val, 4),
        current_price=round(price, 4),
        position=position,
        deviation_pct=round(dev_pct, 2),
        std1_upper=round(vwap_val + std, 4),
        std1_lower=round(vwap_val - std, 4),
        std2_upper=round(vwap_val + 2 * std, 4),
        std2_lower=round(vwap_val - 2 * std, 4),
        bars_used=len(window),
    )


def volume_profile(bars: list[Bar], n_buckets: int = 20) -> VolumeProfileResult | None:
    """Compute Volume Profile: POC, VAH, VAL.

    Divides the price range into n_buckets and assigns volume to each bucket.
    Value Area = price range containing ~70% of all traded volume.
    """
    if len(bars) < 5:
        return None

    symbol = bars[0].symbol
    price_low = min(b.low for b in bars)
    price_high = max(b.high for b in bars)

    if price_high <= price_low:
        return None

    bucket_size = (price_high - price_low) / n_buckets
    if bucket_size <= 0:
        return None

    buckets: list[float] = [0.0] * n_buckets
    for b in bars:
        tp = (b.high + b.low + b.close) / 3.0
        idx = int((tp - price_low) / bucket_size)
        idx = max(0, min(n_buckets - 1, idx))
        buckets[idx] += max(b.volume, 0.0)

    total_volume = sum(buckets)
    if total_volume <= 0:
        return None

    poc_idx = max(range(n_buckets), key=lambda i: buckets[i])
    poc_price = price_low + (poc_idx + 0.5) * bucket_size

    # Expand from POC until 70% of volume is included
    target = total_volume * 0.70
    accumulated = buckets[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target:
        can_up = hi_idx < n_buckets - 1
        can_down = lo_idx > 0
        if not can_up and not can_down:
            break
        up_vol = buckets[hi_idx + 1] if can_up else -1.0
        down_vol = buckets[lo_idx - 1] if can_down else -1.0
        if up_vol >= down_vol:
            hi_idx += 1
            accumulated += buckets[hi_idx]
        else:
            lo_idx -= 1
            accumulated += buckets[lo_idx]

    vah = price_low + (hi_idx + 1) * bucket_size
    val = price_low + lo_idx * bucket_size

    current_price = bars[-1].close
    tol = bucket_size * 0.5
    if current_price > poc_price + tol:
        poc_relation = "above_poc"
    elif current_price < poc_price - tol:
        poc_relation = "below_poc"
    else:
        poc_relation = "at_poc"

    return VolumeProfileResult(
        symbol=symbol,
        poc_price=round(poc_price, 4),
        vah=round(vah, 4),
        val=round(val, 4),
        current_price=round(current_price, 4),
        poc_relation=poc_relation,
        n_buckets=n_buckets,
        bars_used=len(bars),
    )
