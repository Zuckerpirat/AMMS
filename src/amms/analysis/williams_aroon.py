"""Williams %R and Aroon Oscillator.

Two complementary momentum oscillators that measure price position
within recent ranges and trend strength/direction.

Williams %R:
  %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
  Range: -100 (oversold) to 0 (overbought)
  Overbought: %R > -20
  Oversold:   %R < -80

Aroon Oscillator:
  Aroon Up   = ((period - bars since period high) / period) * 100
  Aroon Down = ((period - bars since period low)  / period) * 100
  Aroon Osc  = Aroon Up - Aroon Down  (-100 to +100)
  Strong up trend:   Aroon Up > 70, Aroon Down < 30
  Strong down trend: Aroon Down > 70, Aroon Up < 30

Combined signal uses both indicators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WilliamsAroonReport:
    symbol: str

    # Williams %R
    williams_r: float           # -100 to 0
    williams_overbought: bool   # %R > -20
    williams_oversold: bool     # %R < -80
    williams_signal: str        # "overbought", "neutral", "oversold"

    # Aroon
    aroon_up: float             # 0 to 100
    aroon_down: float           # 0 to 100
    aroon_osc: float            # -100 to +100
    aroon_signal: str           # "strong_up", "up", "neutral", "down", "strong_down"

    # Composite
    composite_signal: str       # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    composite_score: float      # -100 to +100

    # History for trend
    williams_r_series: list[float]  # last 10 values

    current_price: float
    bars_used: int
    verdict: str


def analyze(
    bars: list,
    *,
    symbol: str = "",
    williams_period: int = 14,
    aroon_period: int = 25,
    history: int = 10,
) -> WilliamsAroonReport | None:
    """Compute Williams %R and Aroon Oscillator.

    bars: bar objects with .high, .low, .close attributes.
    """
    min_bars = max(williams_period, aroon_period) + history + 5
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

    # Williams %R at current bar
    hh = max(highs[-williams_period:])
    ll = min(lows[-williams_period:])
    if hh - ll > 1e-9:
        wr = (hh - current) / (hh - ll) * -100.0
    else:
        wr = -50.0
    wr = max(-100.0, min(0.0, wr))

    # Williams %R history (last `history` bars)
    wr_series = []
    for i in range(history, 0, -1):
        idx = n - i
        if idx < williams_period:
            continue
        h_w = max(highs[idx - williams_period:idx])
        l_w = min(lows[idx - williams_period:idx])
        c_w = closes[idx - 1]
        if h_w - l_w > 1e-9:
            wr_series.append((h_w - c_w) / (h_w - l_w) * -100.0)
        else:
            wr_series.append(-50.0)
    wr_series.append(wr)  # include current

    # Williams signals
    w_overbought = wr > -20
    w_oversold   = wr < -80
    if w_overbought:
        w_signal = "overbought"
    elif w_oversold:
        w_signal = "oversold"
    else:
        w_signal = "neutral"

    # Aroon Up/Down at current bar
    # bars since period high: index of max in last aroon_period bars, counting from right
    h_window = highs[-aroon_period:]
    l_window = lows[-aroon_period:]
    max_idx = h_window.index(max(h_window))  # position in window
    min_idx = l_window.index(min(l_window))
    # bars since = aroon_period - 1 - max_idx (0=just happened, period-1=oldest)
    bars_since_high = (aroon_period - 1) - max_idx
    bars_since_low  = (aroon_period - 1) - min_idx

    aroon_up   = (aroon_period - bars_since_high) / aroon_period * 100.0
    aroon_down = (aroon_period - bars_since_low)  / aroon_period * 100.0
    aroon_osc  = aroon_up - aroon_down

    # Aroon signal
    if aroon_up > 70 and aroon_down < 30:
        a_signal = "strong_up"
    elif aroon_up > aroon_down and aroon_osc > 30:
        a_signal = "up"
    elif aroon_down > 70 and aroon_up < 30:
        a_signal = "strong_down"
    elif aroon_down > aroon_up and aroon_osc < -30:
        a_signal = "down"
    else:
        a_signal = "neutral"

    # Composite score: blend Williams (mapped to -100..+100) and Aroon Osc
    # Williams: -100 (oversold=bullish) → +100, -20 (overbought=bearish) → -100
    # Invert: oversold = bullish potential
    w_score = -(wr + 50) * 2.0  # -100 maps to +100, 0 maps to -100
    w_score = max(-100.0, min(100.0, w_score))
    composite = w_score * 0.4 + aroon_osc * 0.6
    composite = max(-100.0, min(100.0, composite))

    if composite >= 60:
        c_signal = "strong_bull"
    elif composite >= 20:
        c_signal = "bull"
    elif composite <= -60:
        c_signal = "strong_bear"
    elif composite <= -20:
        c_signal = "bear"
    else:
        c_signal = "neutral"

    verdict = (
        f"Williams %R + Aroon ({symbol}): {c_signal.replace('_', ' ')}. "
        f"%R={wr:.1f} ({w_signal}), Aroon Osc={aroon_osc:+.1f} ({a_signal.replace('_', ' ')})."
    )

    return WilliamsAroonReport(
        symbol=symbol,
        williams_r=round(wr, 2),
        williams_overbought=w_overbought,
        williams_oversold=w_oversold,
        williams_signal=w_signal,
        aroon_up=round(aroon_up, 1),
        aroon_down=round(aroon_down, 1),
        aroon_osc=round(aroon_osc, 1),
        aroon_signal=a_signal,
        composite_signal=c_signal,
        composite_score=round(composite, 2),
        williams_r_series=[round(v, 2) for v in wr_series],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
