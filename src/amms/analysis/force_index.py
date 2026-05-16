"""Force Index Indicator.

Developed by Alexander Elder. Measures the force (power) of a price move
by combining price direction, magnitude, and volume:

  Raw Force Index = (Close - PrevClose) × Volume

Smoothed versions:
  FI(2)  = EMA(RawFI, 2)   — short-term: identifies short corrections
  FI(13) = EMA(RawFI, 13)  — medium-term: identifies trend changes

Interpretation:
  Positive FI  → bulls in control
  Negative FI  → bears in control
  FI(13) crossing zero → trend change signal
  Large FI spikes → climactic buying/selling

Elder's rules:
  Buy:  FI(2) < 0 during uptrend (correction)
  Sell: FI(2) > 0 during downtrend (rally)
  FI(13) > 0 → uptrend confirmed
  FI(13) < 0 → downtrend confirmed
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForceIndexReport:
    symbol: str

    # Raw Force Index (last bar)
    raw_fi: float

    # Smoothed versions
    fi_2: float      # 2-bar EMA of raw FI (short-term)
    fi_13: float     # 13-bar EMA of raw FI (medium-term)

    # Signals
    fi2_positive: bool
    fi13_positive: bool
    fi13_zero_cross: bool   # FI(13) crossed zero recently

    # Elder's buy/sell context
    buy_setup: bool    # fi2 < 0 and trend is up
    sell_setup: bool   # fi2 > 0 and trend is down
    trend_confirmed: str  # "up", "down", "unclear"

    # Score
    score: float       # -100 to +100
    signal: str        # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # Spike detection
    fi2_spike: bool    # |fi2| > 2× recent average |fi2|

    # History
    fi2_series: list[float]
    fi13_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _ema(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    k = 2.0 / (period + 1)
    val = sum(series[:period]) / period
    result = [val]
    for x in series[period:]:
        val = x * k + val * (1 - k)
        result.append(val)
    return result


def analyze(
    bars: list,
    *,
    symbol: str = "",
    fi_short: int = 2,
    fi_long: int = 13,
    history: int = 15,
    trend_period: int = 20,
) -> ForceIndexReport | None:
    """Compute the Force Index.

    bars: bar objects with .close and .volume attributes.
    """
    min_bars = fi_long + trend_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        volumes = [float(b.volume) for b in bars]
    except (AttributeError, TypeError, ValueError):
        volumes = [1.0] * len(bars)

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Raw Force Index
    raw_fi = []
    for i in range(1, n):
        raw_fi.append((closes[i] - closes[i - 1]) * volumes[i])

    if not raw_fi:
        return None

    # Short and long EMA of raw FI
    fi2_vals  = _ema(raw_fi, fi_short)
    fi13_vals = _ema(raw_fi, fi_long)

    if not fi2_vals or not fi13_vals:
        return None

    cur_fi2  = fi2_vals[-1]
    cur_fi13 = fi13_vals[-1]
    cur_raw  = raw_fi[-1]

    # FI(13) zero cross (last 3 bars)
    zero_cross = False
    if len(fi13_vals) >= 3:
        prev_fi13 = fi13_vals[-2]
        if (cur_fi13 >= 0 and prev_fi13 < 0) or (cur_fi13 <= 0 and prev_fi13 > 0):
            zero_cross = True

    # Trend context from EMA slope
    fi2_pos  = cur_fi2  > 0
    fi13_pos = cur_fi13 > 0

    if len(closes) >= trend_period:
        trend_ema = _ema(closes, trend_period)
        if len(trend_ema) >= 2:
            trend_conf = "up" if trend_ema[-1] > trend_ema[-2] else ("down" if trend_ema[-1] < trend_ema[-2] else "unclear")
        else:
            trend_conf = "unclear"
    else:
        trend_conf = "unclear"

    # Elder's setups
    buy_setup  = not fi2_pos and trend_conf == "up"
    sell_setup = fi2_pos and trend_conf == "down"

    # Spike detection: |fi2| vs recent average
    fi2_recent = fi2_vals[-min(20, len(fi2_vals)):]
    avg_abs_fi2 = sum(abs(v) for v in fi2_recent) / len(fi2_recent) if fi2_recent else 1.0
    fi2_spike = abs(cur_fi2) > avg_abs_fi2 * 2.0

    # Score: normalise fi13 by max abs in window
    fi13_window = fi13_vals[-50:] if len(fi13_vals) >= 50 else fi13_vals
    fi13_max = max(abs(v) for v in fi13_window) if fi13_window else 1.0
    fi13_norm = max(-100.0, min(100.0, cur_fi13 / fi13_max * 100.0)) if fi13_max > 1e-9 else 0.0

    fi2_max = max(abs(v) for v in fi2_recent) if fi2_recent else 1.0
    fi2_norm = max(-100.0, min(100.0, cur_fi2 / fi2_max * 100.0)) if fi2_max > 1e-9 else 0.0

    score = fi13_norm * 0.7 + fi2_norm * 0.3
    score = max(-100.0, min(100.0, score))

    if score >= 50:
        signal = "strong_bull"
    elif score >= 15:
        signal = "bull"
    elif score <= -50:
        signal = "strong_bear"
    elif score <= -15:
        signal = "bear"
    else:
        signal = "neutral"

    # History
    hist_fi2  = fi2_vals[-history:]
    hist_fi13 = fi13_vals[-history:]

    verdict = (
        f"Force Index ({symbol}): {signal.replace('_', ' ')}. "
        f"FI(2)={cur_fi2:+.1f}, FI(13)={cur_fi13:+.1f}, trend: {trend_conf}."
    )
    if buy_setup:
        verdict += " Buy setup (FI2 negative in uptrend)."
    if sell_setup:
        verdict += " Sell setup (FI2 positive in downtrend)."
    if fi2_spike:
        verdict += " Volume spike detected."

    return ForceIndexReport(
        symbol=symbol,
        raw_fi=round(cur_raw, 2),
        fi_2=round(cur_fi2, 2),
        fi_13=round(cur_fi13, 2),
        fi2_positive=fi2_pos,
        fi13_positive=fi13_pos,
        fi13_zero_cross=zero_cross,
        buy_setup=buy_setup,
        sell_setup=sell_setup,
        trend_confirmed=trend_conf,
        score=round(score, 2),
        signal=signal,
        fi2_spike=fi2_spike,
        fi2_series=[round(v, 2) for v in hist_fi2],
        fi13_series=[round(v, 2) for v in hist_fi13],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
