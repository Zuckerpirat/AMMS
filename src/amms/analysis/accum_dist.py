"""Accumulation/Distribution Index (ADI) Analyser.

The A/D Index is a cumulative volume indicator that uses the Money Flow
Multiplier (close position within bar range) to weight volume. Unlike OBV
which adds all volume on up bars, ADI only counts the proportion of volume
that went toward accumulation or distribution.

Formula:
  MFM = ((close - low) - (high - close)) / (high - low)
  MFV = MFM × volume
  ADI = running sum of MFV

Divergences between price and ADI signal potential reversals:
  - Price rising, ADI falling → hidden distribution (bearish)
  - Price falling, ADI rising → hidden accumulation (bullish)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ADIReport:
    symbol: str

    adi: float              # current A/D Index value
    adi_trend: str          # "rising", "falling", "flat"
    adi_roc: float          # rate of change of ADI over lookback

    price_trend: str        # "rising", "falling", "flat"
    price_roc: float        # rate of change of price over lookback

    divergence: str         # "bullish", "bearish", "none"
    divergence_strength: float   # 0-1

    # EMA of ADI (smoothed signal line)
    adi_ema: float
    adi_above_ema: bool

    current_price: float
    cumulative_mfv: float    # total money flow volume
    avg_mfm: float           # average money flow multiplier

    history_adi: list[float]
    bars_used: int
    verdict: str


def _mfm(high: float, low: float, close: float) -> float:
    rng = high - low
    if rng < 1e-9:
        return 0.0
    return ((close - low) - (high - close)) / rng


def _ema_of(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 20,
    ema_period: int = 10,
    history_bars: int = 50,
) -> ADIReport | None:
    """Compute Accumulation/Distribution Index.

    bars: bar objects with .high, .low, .close, .volume attributes.
    lookback: bars for trend/divergence comparison.
    ema_period: smoothing period for ADI signal line.
    history_bars: number of historical ADI values to return.
    """
    if not bars or len(bars) < max(lookback, ema_period) + 5:
        return None

    try:
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

    # Build cumulative ADI
    mfms = [_mfm(highs[i], lows[i], closes[i]) for i in range(n)]
    mfvs = [mfms[i] * volumes[i] for i in range(n)]

    adi_series: list[float] = []
    running = 0.0
    for mfv in mfvs:
        running += mfv
        adi_series.append(running)

    cur_adi = adi_series[-1]

    # ADI trend: compare last bar vs lookback bars ago
    adi_then = adi_series[-lookback] if len(adi_series) > lookback else adi_series[0]
    adi_roc = (cur_adi - adi_then) / (abs(adi_then) + 1e-9) * 100.0

    if adi_roc > 1.0:
        adi_trend = "rising"
    elif adi_roc < -1.0:
        adi_trend = "falling"
    else:
        adi_trend = "flat"

    # Price trend over same lookback
    price_then = closes[-lookback] if len(closes) > lookback else closes[0]
    price_roc = (current - price_then) / price_then * 100.0 if price_then > 0 else 0.0

    if price_roc > 0.5:
        price_trend = "rising"
    elif price_roc < -0.5:
        price_trend = "falling"
    else:
        price_trend = "flat"

    # Divergence detection
    if price_trend == "rising" and adi_trend == "falling":
        divergence = "bearish"   # hidden distribution
        div_strength = min(1.0, abs(adi_roc) / 20.0)
    elif price_trend == "falling" and adi_trend == "rising":
        divergence = "bullish"   # hidden accumulation
        div_strength = min(1.0, abs(adi_roc) / 20.0)
    else:
        divergence = "none"
        div_strength = 0.0

    # EMA of ADI for signal line
    ema_vals = _ema_of(adi_series, ema_period)
    cur_ema = ema_vals[-1] if ema_vals else cur_adi
    adi_above_ema = cur_adi > cur_ema

    # Summary stats
    avg_mfm_val = sum(mfms[-lookback:]) / lookback if len(mfms) >= lookback else sum(mfms) / len(mfms)

    # History
    n_hist = min(history_bars, len(adi_series))
    history = [round(v, 2) for v in adi_series[-n_hist:]]

    # Verdict
    if divergence == "bearish":
        action = f"Bearish divergence — price rising but ADI falling (hidden distribution)"
    elif divergence == "bullish":
        action = f"Bullish divergence — price falling but ADI rising (hidden accumulation)"
    elif adi_trend == "rising":
        action = "ADI rising — accumulation in control"
    elif adi_trend == "falling":
        action = "ADI falling — distribution in control"
    else:
        action = "ADI flat — no clear pressure"

    ema_tag = "above" if adi_above_ema else "below"
    verdict = f"A/D Index: {action}. ADI {ema_tag} EMA-{ema_period}. Price ROC={price_roc:+.1f}%."

    return ADIReport(
        symbol=symbol,
        adi=round(cur_adi, 2),
        adi_trend=adi_trend,
        adi_roc=round(adi_roc, 2),
        price_trend=price_trend,
        price_roc=round(price_roc, 2),
        divergence=divergence,
        divergence_strength=round(div_strength, 3),
        adi_ema=round(cur_ema, 2),
        adi_above_ema=adi_above_ema,
        current_price=round(current, 4),
        cumulative_mfv=round(cur_adi, 2),
        avg_mfm=round(avg_mfm_val, 4),
        history_adi=history,
        bars_used=n,
        verdict=verdict,
    )
