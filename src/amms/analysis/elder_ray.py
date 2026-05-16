"""Elder Ray Index.

Developed by Alexander Elder. Uses a 13-bar EMA as the "force" reference and
measures:

  Bull Power = High - EMA   (positive = bulls pushing price above EMA)
  Bear Power = Low  - EMA   (negative = bears pushing price below EMA)

Trading signals:
  - Buy:  Bear Power negative but rising + Bull Power positive
  - Sell: Bull Power positive but falling + Bear Power negative
  - Strong bull: both positive
  - Strong bear: both negative

Also computes a normalised Elder Ray score (-100..+100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ElderRayReport:
    symbol: str

    # EMA
    ema: float
    ema_period: int

    # Elder Ray components
    bull_power: float   # High - EMA
    bear_power: float   # Low  - EMA

    # Trend of each component (last N bars)
    bull_rising: bool   # bull_power trending up
    bear_rising: bool   # bear_power trending toward 0 (less negative = rising)

    # Score
    score: float        # -100 to +100
    signal: str         # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # Series for context (last 10 bars)
    bull_series: list[float]
    bear_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _ema_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    result = [ema]
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
        result.append(ema)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    ema_period: int = 13,
    trend_lookback: int = 5,
    history: int = 10,
) -> ElderRayReport | None:
    """Compute Elder Ray Index (Bull Power and Bear Power).

    bars: bar objects with .high, .low, .close attributes.
    """
    min_bars = ema_period + trend_lookback + history + 3
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # EMA series over all closes
    ema_vals = _ema_series(closes, ema_period)
    if not ema_vals:
        return None

    # Align: ema_vals[i] corresponds to closes[ema_period - 1 + i]
    # Bull/Bear Power series
    bull_series_full = []
    bear_series_full = []
    for i, ema_v in enumerate(ema_vals):
        bar_idx = ema_period - 1 + i
        bull_series_full.append(highs[bar_idx] - ema_v)
        bear_series_full.append(lows[bar_idx] - ema_v)

    current_ema = ema_vals[-1]
    current_bull = highs[-1] - current_ema
    current_bear = lows[-1] - current_ema

    # Recent trend: is bull power rising/falling over last trend_lookback bars?
    if len(bull_series_full) >= trend_lookback + 1:
        bull_trend = bull_series_full[-1] - bull_series_full[-trend_lookback - 1]
        bear_trend = bear_series_full[-1] - bear_series_full[-trend_lookback - 1]
        bull_rising = bull_trend > 0
        bear_rising = bear_trend > 0  # rising toward 0 = less bearish
    else:
        bull_rising = False
        bear_rising = False

    # History for output (last `history` bars)
    hist_bull = bull_series_full[-history:]
    hist_bear = bear_series_full[-history:]

    # Score computation
    # Normalise by ATR proxy: use average of abs(bull) + abs(bear) over history
    atr_proxy = (
        sum(abs(b) + abs(be) for b, be in zip(hist_bull, hist_bear)) / len(hist_bull)
        if hist_bull else 1.0
    )
    if atr_proxy < 1e-9:
        atr_proxy = 1.0

    bull_score = current_bull / atr_proxy * 50.0
    bear_score = current_bear / atr_proxy * 50.0
    score = bull_score + bear_score  # bear is usually negative, dampens score
    score = max(-100.0, min(100.0, score))

    # Signal
    if current_bull > 0 and current_bear > 0:
        signal = "strong_bull"
    elif current_bull < 0 and current_bear < 0:
        signal = "strong_bear"
    elif current_bear < 0 and bear_rising and current_bull > 0:
        signal = "bull"   # bear power improving, bulls in control
    elif current_bull > 0 and not bull_rising and current_bear < 0:
        signal = "bear"   # bull power fading
    elif score >= 20:
        signal = "bull"
    elif score <= -20:
        signal = "bear"
    else:
        signal = "neutral"

    verdict = (
        f"Elder Ray ({symbol}, EMA-{ema_period}): {signal.replace('_', ' ')}. "
        f"Bull Power {current_bull:+.3f}, Bear Power {current_bear:+.3f}. "
        f"EMA={current_ema:.2f}."
    )

    return ElderRayReport(
        symbol=symbol,
        ema=round(current_ema, 4),
        ema_period=ema_period,
        bull_power=round(current_bull, 4),
        bear_power=round(current_bear, 4),
        bull_rising=bull_rising,
        bear_rising=bear_rising,
        score=round(score, 2),
        signal=signal,
        bull_series=[round(v, 4) for v in hist_bull],
        bear_series=[round(v, 4) for v in hist_bear],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
