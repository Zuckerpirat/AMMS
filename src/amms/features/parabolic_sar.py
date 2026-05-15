"""Parabolic SAR (Stop and Reverse).

A trailing stop indicator that follows price closely during a trend.

When price is above SAR → uptrend (long signal).
When price falls below SAR → reversal to downtrend.

Parameters:
  acceleration (AF): starts at initial_af (0.02), increments by step (0.02)
  each time a new extreme is set, up to max_af (0.20).

SAR formula:
  Uptrend:   SAR(n) = SAR(n-1) + AF * (EP - SAR(n-1))
  Downtrend: SAR(n) = SAR(n-1) + AF * (EP - SAR(n-1))
  EP = extreme point (highest high in uptrend, lowest low in downtrend)
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class SARResult:
    sar: float          # Current SAR value
    trend: str          # "up" | "down"
    acceleration: float  # Current AF value
    extreme: float      # Current extreme point
    distance_pct: float  # % distance of price from SAR (absolute)


def parabolic_sar(
    bars: list[Bar],
    initial_af: float = 0.02,
    step: float = 0.02,
    max_af: float = 0.20,
) -> SARResult | None:
    """Compute the current Parabolic SAR.

    Returns None if fewer than 2 bars.
    The SAR for the most recent bar is returned.
    """
    if len(bars) < 2:
        return None

    # Initialize with first two bars
    if bars[1].close >= bars[0].close:
        trend = "up"
        sar = bars[0].low
        ep = bars[1].high
    else:
        trend = "down"
        sar = bars[0].high
        ep = bars[1].low

    af = initial_af

    for i in range(2, len(bars)):
        prev_bar = bars[i - 1]
        curr_bar = bars[i]

        # Compute new SAR
        new_sar = sar + af * (ep - sar)

        if trend == "up":
            # SAR must not be above the two previous lows
            new_sar = min(new_sar, prev_bar.low, bars[i - 2].low if i >= 2 else prev_bar.low)
            if curr_bar.low < new_sar:
                # Reversal to downtrend
                trend = "down"
                sar = ep  # SAR switches to the highest high
                ep = curr_bar.low
                af = initial_af
            else:
                sar = new_sar
                if curr_bar.high > ep:
                    ep = curr_bar.high
                    af = min(af + step, max_af)
        else:  # downtrend
            # SAR must not be below the two previous highs
            new_sar = max(new_sar, prev_bar.high, bars[i - 2].high if i >= 2 else prev_bar.high)
            if curr_bar.high > new_sar:
                # Reversal to uptrend
                trend = "up"
                sar = ep  # SAR switches to the lowest low
                ep = curr_bar.high
                af = initial_af
            else:
                sar = new_sar
                if curr_bar.low < ep:
                    ep = curr_bar.low
                    af = min(af + step, max_af)

    price = bars[-1].close
    distance_pct = abs(price - sar) / price * 100 if price > 0 else 0.0

    return SARResult(
        sar=round(sar, 4),
        trend=trend,
        acceleration=round(af, 4),
        extreme=round(ep, 4),
        distance_pct=round(distance_pct, 2),
    )
