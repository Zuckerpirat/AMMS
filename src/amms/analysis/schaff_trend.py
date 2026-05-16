"""Schaff Trend Cycle (STC) Analyser.

The STC combines MACD with a double Stochastic smoothing pass to produce a
fast, low-lag oscillator bounded 0–100.

Algorithm (Pring / Schaff):
  1. MACD = EMA(close, fast) - EMA(close, slow)
  2. First  Stochastic of MACD over `cycle` bars → K1 → smooth to D1 (EMA)
  3. Second Stochastic of D1  over `cycle` bars → K2 → smooth to D2 (EMA) = STC

Signal:
  - STC rising above 25  → buy trigger
  - STC falling below 75 → sell trigger
  - STC > 75             → overbought
  - STC < 25             → oversold
  - Slope (vs 3 bars ago) used for momentum direction
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class STCReport:
    symbol: str

    stc: float          # 0-100, current value
    stc_prev: float     # value 1 bar ago (for cross detection)
    macd: float         # underlying MACD

    overbought: bool    # stc > 75
    oversold: bool      # stc < 25
    buy_trigger: bool   # crossed above 25 within last 3 bars
    sell_trigger: bool  # crossed below 75 within last 3 bars
    slope_up: bool      # stc rising (vs 3 bars ago)

    score: float        # -100..+100
    signal: str         # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    history: list[float]   # recent STC values
    bars_used: int
    verdict: str


def _ema(values: list[float], period: int) -> list[float]:
    """Return EMA series (same length as input, warm-up via SMA)."""
    if not values or period < 1:
        return []
    out: list[float] = []
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out.append(seed)
    for v in values[period:]:
        out.append(out[-1] + k * (v - out[-1]))
    # Pad front with None placeholders then fill — keep length by returning
    # only the warm part: len = len(values) - period + 1
    return out


def _stochastic_of(series: list[float], period: int) -> list[float]:
    """Raw Stochastic %K of an arbitrary series."""
    out: list[float] = []
    for i in range(len(series)):
        if i < period - 1:
            out.append(50.0)
            continue
        window = series[i - period + 1 : i + 1]
        lo = min(window)
        hi = max(window)
        rng = hi - lo
        if rng < 1e-9:
            out.append(out[-1] if out else 50.0)
        else:
            out.append((series[i] - lo) / rng * 100.0)
    return out


def _smooth_ema(series: list[float], period: int = 3) -> list[float]:
    """Fast EMA smoothing of a series (for Stochastic D lines)."""
    if not series:
        return []
    k = 2.0 / (period + 1)
    out = [series[0]]
    for v in series[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def analyze(
    bars: list,
    *,
    symbol: str = "",
    fast: int = 23,
    slow: int = 50,
    cycle: int = 10,
    smooth: int = 3,
    history: int = 20,
) -> STCReport | None:
    """Compute Schaff Trend Cycle.

    bars: bar objects with .close attribute.
    fast / slow: MACD EMA periods.
    cycle: Stochastic lookback for double-smoothing.
    smooth: EMA period for D lines.
    Requires slow + cycle + smooth + history bars minimum.
    """
    min_bars = slow + cycle + smooth + history
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(closes)

    # Step 1: MACD line
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)

    # Align: ema_fast has length n-fast+1, ema_slow has length n-slow+1
    # MACD starts where the slower EMA starts
    macd_len = len(ema_slow)
    fast_offset = len(ema_fast) - macd_len  # how many extra fast values at start
    macd = [ema_fast[fast_offset + i] - ema_slow[i] for i in range(macd_len)]

    if len(macd) < cycle + smooth + history:
        return None

    # Step 2: First Stochastic pass over MACD
    k1 = _stochastic_of(macd, cycle)
    d1 = _smooth_ema(k1, smooth)

    # Step 3: Second Stochastic pass over D1
    k2 = _stochastic_of(d1, cycle)
    stc_series = _smooth_ema(k2, smooth)

    if len(stc_series) < history + 4:
        return None

    recent_stc = stc_series[-(history):]
    stc_val   = stc_series[-1]
    stc_prev  = stc_series[-2]
    stc_3ago  = stc_series[-4] if len(stc_series) >= 4 else stc_series[0]

    macd_val = macd[-1]

    overbought = stc_val > 75.0
    oversold   = stc_val < 25.0
    slope_up   = stc_val > stc_3ago

    # Cross detection (within last 3 bars)
    buy_trigger  = False
    sell_trigger = False
    window = stc_series[-4:] if len(stc_series) >= 4 else stc_series
    for i in range(1, len(window)):
        if window[i - 1] < 25.0 and window[i] >= 25.0:
            buy_trigger = True
        if window[i - 1] > 75.0 and window[i] <= 75.0:
            sell_trigger = True

    # Score: position within 0-100 range re-scaled to -100..+100,
    # then adjusted for slope
    pos_score = (stc_val - 50.0) * 2.0   # -100..+100
    slope_boost = 15.0 if slope_up else -15.0
    score = max(-100.0, min(100.0, pos_score + slope_boost * 0.5))

    if overbought and slope_up:
        signal = "strong_buy"
    elif score >= 40 or (buy_trigger and slope_up):
        signal = "buy"
    elif score <= -40 or (sell_trigger and not slope_up):
        signal = "sell"
    elif stc_val < 15.0 or (oversold and not slope_up):
        signal = "strong_sell"
    else:
        signal = "neutral"

    trend_word = "rising" if slope_up else "falling"
    ob_str = " [OVERBOUGHT]" if overbought else (" [OVERSOLD]" if oversold else "")
    trigger_str = ""
    if buy_trigger:
        trigger_str = " Buy trigger (crossed 25)."
    elif sell_trigger:
        trigger_str = " Sell trigger (crossed 75)."

    verdict = (
        f"STC ({symbol}): {stc_val:.1f}/100{ob_str}, {trend_word}. "
        f"MACD {macd_val:+.4f}. Signal: {signal.replace('_', ' ')}.{trigger_str}"
    )

    return STCReport(
        symbol=symbol,
        stc=round(stc_val, 2),
        stc_prev=round(stc_prev, 2),
        macd=round(macd_val, 6),
        overbought=overbought,
        oversold=oversold,
        buy_trigger=buy_trigger,
        sell_trigger=sell_trigger,
        slope_up=slope_up,
        score=round(score, 2),
        signal=signal,
        history=[round(v, 2) for v in recent_stc],
        bars_used=n,
        verdict=verdict,
    )
