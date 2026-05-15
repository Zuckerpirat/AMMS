"""Ichimoku Cloud (Ichimoku Kinko Hyo).

Five lines that together describe trend, momentum, and support/resistance:

  Tenkan-sen (Conversion):  (highest_high + lowest_low) / 2  over 9 periods
  Kijun-sen (Base):         (highest_high + lowest_low) / 2  over 26 periods
  Senkou Span A (Leading A): (Tenkan + Kijun) / 2  shifted +26 periods
  Senkou Span B (Leading B): (highest_high + lowest_low) / 2 over 52 periods, shifted +26
  Chikou Span (Lagging):    current close shifted back 26 periods

The "cloud" (kumo) is the area between Span A and Span B.
Price above cloud = bullish. Price below cloud = bearish. Inside = neutral.

Tenkan > Kijun = short-term bullish momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class IchimokuResult:
    tenkan: float          # Conversion line (9-period mid)
    kijun: float           # Base line (26-period mid)
    span_a: float          # Leading Span A (current, not shifted)
    span_b: float          # Leading Span B (52-period mid, current)
    price: float           # Last close
    cloud_top: float       # max(span_a, span_b)
    cloud_bottom: float    # min(span_a, span_b)
    position: str          # "above_cloud" | "in_cloud" | "below_cloud"
    momentum: str          # "bullish" | "bearish" | "neutral" (tenkan vs kijun)
    cloud_color: str       # "green" (span_a > span_b) | "red" | "flat"


def _midpoint(bars: list[Bar], n: int) -> float | None:
    if len(bars) < n:
        return None
    window = bars[-n:]
    return (max(b.high for b in window) + min(b.low for b in window)) / 2.0


def ichimoku(
    bars: list[Bar],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> IchimokuResult | None:
    """Compute current Ichimoku values.

    Returns None if insufficient data (needs at least senkou_b_period bars).
    """
    if len(bars) < senkou_b_period:
        return None

    tenkan = _midpoint(bars, tenkan_period)
    kijun = _midpoint(bars, kijun_period)
    span_b = _midpoint(bars, senkou_b_period)

    if tenkan is None or kijun is None or span_b is None:
        return None

    span_a = (tenkan + kijun) / 2.0
    price = bars[-1].close
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    if price > cloud_top:
        position = "above_cloud"
    elif price < cloud_bottom:
        position = "below_cloud"
    else:
        position = "in_cloud"

    if tenkan > kijun:
        momentum = "bullish"
    elif tenkan < kijun:
        momentum = "bearish"
    else:
        momentum = "neutral"

    if span_a > span_b + 0.01:
        cloud_color = "green"
    elif span_b > span_a + 0.01:
        cloud_color = "red"
    else:
        cloud_color = "flat"

    return IchimokuResult(
        tenkan=round(tenkan, 4),
        kijun=round(kijun, 4),
        span_a=round(span_a, 4),
        span_b=round(span_b, 4),
        price=round(price, 4),
        cloud_top=round(cloud_top, 4),
        cloud_bottom=round(cloud_bottom, 4),
        position=position,
        momentum=momentum,
        cloud_color=cloud_color,
    )
