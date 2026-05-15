"""Commodity Channel Index (CCI).

CCI = (Typical Price - SMA(n)) / (0.015 * Mean Absolute Deviation)

Where Typical Price = (high + low + close) / 3

Range: typically -200..+200 (can exceed)
  CCI > +100  → overbought / strong uptrend momentum
  CCI < -100  → oversold / strong downtrend momentum
  CCI crosses 0 → potential trend change

The 0.015 constant scales ~70-80% of values to fall within ±100.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class CCIResult:
    value: float        # current CCI
    zone: str           # "overbought" | "oversold" | "neutral"
    signal: str         # "buy" | "sell" | "none"
    period: int


def cci(bars: list[Bar], period: int = 20) -> CCIResult | None:
    """Compute Commodity Channel Index for the last bar.

    Returns None if fewer than period bars available.
    """
    if len(bars) < period:
        return None

    window = bars[-period:]
    tps = [(b.high + b.low + b.close) / 3.0 for b in window]
    mean_tp = sum(tps) / period
    mad = sum(abs(tp - mean_tp) for tp in tps) / period

    if mad == 0:
        cci_val = 0.0
    else:
        cci_val = (tps[-1] - mean_tp) / (0.015 * mad)

    if cci_val > 100:
        zone = "overbought"
        signal = "sell"
    elif cci_val < -100:
        zone = "oversold"
        signal = "buy"
    else:
        zone = "neutral"
        signal = "none"

    return CCIResult(
        value=round(cci_val, 2),
        zone=zone,
        signal=signal,
        period=period,
    )
