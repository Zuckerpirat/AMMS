"""Adaptive RSI (ARSI) Analyser.

RSI with a dynamically adjusted period driven by the Kaufman Efficiency
Ratio (ER). In trending markets the period shrinks (faster reaction); in
choppy markets it expands (noise filter).

Algorithm:
  ER  = |Close[t] - Close[t-n]| / sum(|Close[i] - Close[i-1]|, n)
  period(t) = round(fast + (1 - ER) * (slow - fast))
              bounded to [fast, slow]
  ARSI at each bar = RSI(closes[:t+1], period(t))

Also computes:
  - Wilder-smoothed ARSI signal line (EMA approx)
  - Overbought / oversold flags
  - Divergence (price direction vs ARSI direction)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ARSIReport:
    symbol: str

    arsi: float         # current adaptive RSI value (0-100)
    arsi_signal: float  # EMA(ARSI, signal_period)
    er: float           # current Efficiency Ratio (0-1)
    effective_period: int  # RSI period used for the current bar

    overbought: bool    # arsi > ob_level
    oversold: bool      # arsi < os_level
    bullish: bool       # arsi above signal
    cross_up: bool      # just crossed above signal
    cross_down: bool    # just crossed below signal

    price_direction: str   # "up", "down", "flat"
    arsi_direction: str    # "up", "down", "flat"
    divergence: bool

    score: float        # -100..+100
    signal: str         # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    history: list[float]
    bars_used: int
    verdict: str


def _er(closes: list[float], period: int) -> float:
    """Kaufman Efficiency Ratio for the most recent `period` bars."""
    if len(closes) <= period:
        return 0.5
    net = abs(closes[-1] - closes[-period - 1])
    noise = sum(abs(closes[i] - closes[i - 1]) for i in range(-period, 0))
    if noise < 1e-9:
        return 0.0
    return min(1.0, net / noise)


def _rsi(closes: list[float], period: int) -> float:
    """Simple RSI (non-Wilder, using last `period` diffs)."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    if avg_l < 1e-9:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def analyze(
    bars: list,
    *,
    symbol: str = "",
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    signal_period: int = 9,
    ob_level: float = 70.0,
    os_level: float = 30.0,
    history: int = 20,
) -> ARSIReport | None:
    """Compute Adaptive RSI driven by Efficiency Ratio.

    bars: bar objects with .close attribute.
    er_period: lookback for Efficiency Ratio computation.
    fast_period / slow_period: RSI period bounds.
    Requires slow_period + er_period + signal_period + history bars minimum.
    """
    min_bars = slow_period + er_period + signal_period + history
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

    # Compute ARSI series
    arsi_series: list[float] = []
    period_series: list[int] = []
    er_series: list[float] = []

    for t in range(slow_period, n):
        er_val = _er(closes[: t + 1], er_period)
        eff_period = round(fast_period + (1.0 - er_val) * (slow_period - fast_period))
        eff_period = max(fast_period, min(slow_period, eff_period))
        arsi_val = _rsi(closes[: t + 1], eff_period)
        arsi_series.append(arsi_val)
        period_series.append(eff_period)
        er_series.append(er_val)

    if len(arsi_series) < signal_period + history + 2:
        return None

    # Signal line: EMA of ARSI
    k = 2.0 / (signal_period + 1)
    sig = [arsi_series[0]]
    for v in arsi_series[1:]:
        sig.append(sig[-1] + k * (v - sig[-1]))

    arsi_val   = arsi_series[-1]
    arsi_prev  = arsi_series[-2]
    arsi_4ago  = arsi_series[-5] if len(arsi_series) >= 5 else arsi_series[0]
    sig_val    = sig[-1]
    sig_prev   = sig[-2]
    er_val     = er_series[-1]
    eff_period = period_series[-1]

    overbought = arsi_val > ob_level
    oversold   = arsi_val < os_level
    bullish    = arsi_val > sig_val

    cross_up   = arsi_prev <= sig_prev and arsi_val > sig_val
    cross_down = arsi_prev >= sig_prev and arsi_val < sig_val

    # Directions
    start_price = closes[-history] if history < n else closes[0]
    if current > start_price * 1.003:
        price_dir = "up"
    elif current < start_price * 0.997:
        price_dir = "down"
    else:
        price_dir = "flat"

    arsi_start = arsi_series[-(history)] if history < len(arsi_series) else arsi_series[0]
    if arsi_val > arsi_start + 2:
        arsi_dir = "up"
    elif arsi_val < arsi_start - 2:
        arsi_dir = "down"
    else:
        arsi_dir = "flat"

    divergence = (
        (price_dir == "up"   and arsi_dir == "down") or
        (price_dir == "down" and arsi_dir == "up")
    )

    # Score: RSI position + cross bonus
    pos_score = (arsi_val - 50.0) * 2.0
    signal_bonus = 10.0 if bullish else -10.0
    score = max(-100.0, min(100.0, pos_score + signal_bonus))

    if score >= 60:
        sig_label = "strong_buy"
    elif score >= 25:
        sig_label = "buy"
    elif score <= -60:
        sig_label = "strong_sell"
    elif score <= -25:
        sig_label = "sell"
    else:
        sig_label = "neutral"

    cross_str = ""
    if cross_up:
        cross_str = " Bullish cross."
    elif cross_down:
        cross_str = " Bearish cross."

    verdict = (
        f"ARSI ({symbol}): {arsi_val:.1f} (ER={er_val:.2f}, period={eff_period}). "
        f"Signal: {sig_label.replace('_', ' ')}.{cross_str}"
    )
    if divergence:
        verdict += f" Divergence: price {price_dir}, ARSI {arsi_dir}."

    recent_history = arsi_series[-history:]

    return ARSIReport(
        symbol=symbol,
        arsi=round(arsi_val, 2),
        arsi_signal=round(sig_val, 2),
        er=round(er_val, 3),
        effective_period=eff_period,
        overbought=overbought,
        oversold=oversold,
        bullish=bullish,
        cross_up=cross_up,
        cross_down=cross_down,
        price_direction=price_dir,
        arsi_direction=arsi_dir,
        divergence=divergence,
        score=round(score, 2),
        signal=sig_label,
        history=[round(v, 2) for v in recent_history],
        bars_used=n,
        verdict=verdict,
    )
