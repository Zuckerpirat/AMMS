"""Bollinger Band Squeeze Detector.

Detects Bollinger Band squeezes — periods where the band width drops to a
local minimum, indicating compressed volatility that often precedes a
directional breakout.

Uses two complementary indicators:
  1. BandWidth = (Upper - Lower) / Middle  (normalized)
  2. Keltner Channel comparison: BB inside KC = "true squeeze"

The squeeze score ranges 0-100:
  - 0   = bands at maximum width (high vol, no squeeze)
  - 100 = tightest squeeze ever seen in the lookback window

Reports:
  - Current BandWidth and its percentile in the lookback window
  - Whether a squeeze is active (BW < 20th percentile)
  - Direction bias: where price sits within the band
  - Momentum indicator (rate of change since squeeze started)
  - Recent squeeze history
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBSqueeze:
    bar_idx: int
    bandwidth: float       # (upper - lower) / middle
    squeeze_pct: float     # percentile rank of BW (0=tightest, 100=widest)
    is_squeezed: bool      # BW in bottom 20th percentile
    upper: float
    middle: float          # SMA-20
    lower: float
    price: float
    price_position: float  # 0=at lower, 0.5=at middle, 1=at upper


@dataclass(frozen=True)
class BBSqueezeReport:
    symbol: str
    current_bandwidth: float
    bandwidth_percentile: float   # 0=tightest ever (now), 100=widest
    current_squeeze_score: float  # 100 - bandwidth_percentile
    is_squeezed: bool
    squeeze_active_bars: int      # how many consecutive bars in squeeze
    direction_bias: str           # "bullish" / "bearish" / "neutral"
    upper: float
    middle: float
    lower: float
    current_price: float
    price_position: float
    roc_since_squeeze: float      # ROC% from when squeeze started
    history: list[BBSqueeze]      # last N squeeze snapshots
    bars_used: int
    verdict: str


def _sma(vals: list[float], period: int) -> float | None:
    if len(vals) < period:
        return None
    return sum(vals[-period:]) / period


def _stddev(vals: list[float], period: int) -> float:
    if len(vals) < period:
        return 0.0
    subset = vals[-period:]
    mean = sum(subset) / period
    return (sum((v - mean) ** 2 for v in subset) / period) ** 0.5


def analyze(bars: list, *, symbol: str = "", period: int = 20, mult: float = 2.0,
            history_bars: int = 50) -> BBSqueezeReport | None:
    """Detect Bollinger Band squeeze from bar history.

    bars: bar objects with .close, .high, .low attributes.
    period: BB/SMA period (default 20).
    mult: standard deviation multiplier (default 2.0).
    history_bars: how many past bars to include in history.
    """
    min_bars = period + history_bars + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
    except Exception:
        return None

    n = len(closes)

    # Compute BandWidth for all bars from period onward
    bandwidths: list[float] = []
    for i in range(period - 1, n):
        subset = closes[i - period + 1:i + 1]
        mean = sum(subset) / period
        std = (sum((v - mean) ** 2 for v in subset) / period) ** 0.5
        bw = (2 * mult * std) / mean if mean > 0 else 0.0
        bandwidths.append(bw)

    if not bandwidths:
        return None

    # Percentile of current BW in recent lookback
    lookback = min(len(bandwidths), history_bars)
    recent_bw = bandwidths[-lookback:]
    current_bw = bandwidths[-1]
    below = sum(1 for b in recent_bw if b <= current_bw)
    bw_percentile = below / len(recent_bw) * 100.0  # 0=tightest
    squeeze_score = 100.0 - bw_percentile

    threshold_pct = 20.0
    is_squeezed = bw_percentile <= threshold_pct

    # Count consecutive squeezed bars
    squeeze_active = 0
    threshold_val = sorted(recent_bw)[int(len(recent_bw) * threshold_pct / 100)]
    for bw in reversed(bandwidths):
        if bw <= threshold_val:
            squeeze_active += 1
        else:
            break

    # Current BB values
    subset = closes[-period:]
    mean = sum(subset) / period
    std = (sum((v - mean) ** 2 for v in subset) / period) ** 0.5
    upper = mean + mult * std
    lower = mean - mult * std
    price = closes[-1]
    band_range = upper - lower
    price_pos = (price - lower) / band_range if band_range > 0 else 0.5

    # Direction bias from price position
    if price_pos > 0.65:
        direction = "bullish"
    elif price_pos < 0.35:
        direction = "bearish"
    else:
        direction = "neutral"

    # ROC since squeeze started
    if squeeze_active > 0 and len(closes) > squeeze_active:
        squeeze_start_price = closes[-(squeeze_active + 1)]
        roc_squeeze = (price - squeeze_start_price) / squeeze_start_price * 100.0 if squeeze_start_price > 0 else 0.0
    else:
        roc_squeeze = 0.0

    # History: last history_bars BBSqueeze snapshots
    history: list[BBSqueeze] = []
    for idx in range(max(0, len(bandwidths) - history_bars), len(bandwidths)):
        bar_bw = bandwidths[idx]
        recent_for_pct = bandwidths[max(0, idx - lookback + 1):idx + 1]
        b_below = sum(1 for x in recent_for_pct if x <= bar_bw)
        b_pct = b_below / len(recent_for_pct) * 100.0 if recent_for_pct else 50.0
        s_pct = 100.0 - b_pct

        # Absolute bar index in original bars array
        abs_idx = idx + period - 1
        if abs_idx >= n:
            continue
        c = closes[abs_idx]
        # Compute BB for this bar
        sub = closes[abs_idx - period + 1:abs_idx + 1]
        m = sum(sub) / period
        sd = (sum((v - m) ** 2 for v in sub) / period) ** 0.5
        u = m + mult * sd
        lo = m - mult * sd
        br = u - lo
        pp = (c - lo) / br if br > 0 else 0.5

        history.append(BBSqueeze(
            bar_idx=abs_idx,
            bandwidth=round(bar_bw, 5),
            squeeze_pct=round(b_pct, 1),
            is_squeezed=b_pct <= threshold_pct,
            upper=round(u, 4),
            middle=round(m, 4),
            lower=round(lo, 4),
            price=round(c, 4),
            price_position=round(pp, 3),
        ))

    # Verdict
    parts = []
    if is_squeezed:
        parts.append(f"SQUEEZE active ({squeeze_active} bars, score {squeeze_score:.0f}/100)")
    else:
        parts.append(f"no squeeze (BW percentile {bw_percentile:.0f}%)")
    parts.append(f"bias: {direction} (price at {price_pos * 100:.0f}% of band)")
    if is_squeezed and abs(roc_squeeze) > 0.3:
        parts.append(f"drift since squeeze start {roc_squeeze:+.2f}%")

    verdict = "BB squeeze: " + "; ".join(parts) + "."

    return BBSqueezeReport(
        symbol=symbol,
        current_bandwidth=round(current_bw, 5),
        bandwidth_percentile=round(bw_percentile, 1),
        current_squeeze_score=round(squeeze_score, 1),
        is_squeezed=is_squeezed,
        squeeze_active_bars=squeeze_active,
        direction_bias=direction,
        upper=round(upper, 4),
        middle=round(mean, 4),
        lower=round(lower, 4),
        current_price=round(price, 4),
        price_position=round(price_pos, 3),
        roc_since_squeeze=round(roc_squeeze, 3),
        history=history,
        bars_used=len(bars),
        verdict=verdict,
    )
