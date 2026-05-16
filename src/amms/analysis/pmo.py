"""Price Momentum Oscillator (PMO).

Developed by Carl Swenlin for StockCharts.com. A double-smoothed Rate of
Change oscillator that filters noise better than MACD.

Calculation:
  1. ROC = ((Close / Close[1]) - 1) * 100  (1-period % change)
  2. Smoothed ROC (SROC) = EMA(ROC, roc_period) * (roc_period / 10)
  3. PMO = EMA(SROC, smooth_period)
  4. Signal = EMA(PMO, signal_period)
  5. Histogram = PMO - Signal

Interpretation:
  PMO > 0 → bullish momentum
  PMO < 0 → bearish momentum
  PMO crossing signal → trade signal
  PMO overbought (> threshold): potential reversal zone
  PMO oversold  (< -threshold): potential reversal zone

Default periods: ROC=35, Smooth=20, Signal=10 (Swenlin's standard settings)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PMOReport:
    symbol: str

    pmo: float              # current PMO value
    pmo_signal: float       # signal line
    pmo_histogram: float    # PMO - Signal

    pmo_bullish: bool       # PMO > 0
    above_signal: bool      # PMO > signal

    cross_up: bool          # PMO crossed above signal
    cross_down: bool        # PMO crossed below signal

    # Overbought/oversold (relative to own history)
    pmo_pct_rank: float     # 0-100 percentile of PMO in recent window
    overbought: bool        # pct_rank > 80
    oversold: bool          # pct_rank < 20

    score: float            # -100 to +100
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # History for sparkline (last 15 values)
    pmo_series: list[float]
    signal_series: list[float]

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


def _pct_rank(series: list[float], value: float) -> float:
    if not series:
        return 50.0
    return sum(1 for v in series if v <= value) / len(series) * 100.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    roc_period: int = 35,
    smooth_period: int = 20,
    signal_period: int = 10,
    history: int = 15,
) -> PMOReport | None:
    """Compute the Price Momentum Oscillator.

    bars: bar objects with .close attribute.
    Uses Swenlin's standard settings by default (35/20/10).
    """
    # Need: roc_period + smooth_period + signal_period + history + 5
    min_bars = roc_period + smooth_period + signal_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # Step 1: 1-period % ROC
    roc1 = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 1e-9:
            roc1.append((closes[i] / closes[i - 1] - 1.0) * 100.0)
        else:
            roc1.append(0.0)

    # Step 2: EMA of ROC (scaled)
    sroc_raw = _ema(roc1, roc_period)
    if not sroc_raw:
        return None
    sroc = [v * (roc_period / 10.0) for v in sroc_raw]

    # Step 3: PMO = EMA of SROC
    pmo_vals = _ema(sroc, smooth_period)
    if len(pmo_vals) < signal_period + 2:
        return None

    # Step 4: Signal = EMA of PMO
    sig_vals = _ema(pmo_vals, signal_period)
    if not sig_vals:
        return None

    cur_pmo = pmo_vals[-1]
    cur_sig = sig_vals[-1]
    hist    = cur_pmo - cur_sig

    # Cross detection
    if len(pmo_vals) >= 2 and len(sig_vals) >= 2:
        prev_pmo = pmo_vals[-2]
        prev_sig = sig_vals[-2]
        cross_up   = cur_pmo >= cur_sig and prev_pmo < prev_sig
        cross_down = cur_pmo <= cur_sig and prev_pmo > prev_sig
    else:
        cross_up = cross_down = False

    # PMO percentile rank over recent window
    pmo_window = pmo_vals[-50:] if len(pmo_vals) >= 50 else pmo_vals
    pmo_rank = _pct_rank(pmo_window, cur_pmo)
    overbought = pmo_rank > 80
    oversold   = pmo_rank < 20

    # Score
    # Component 1: PMO vs 0 (bullish/bearish), normalised
    pmo_norm_val = max(abs(v) for v in pmo_window) if pmo_window else 1.0
    if pmo_norm_val < 1e-9:
        pmo_norm_val = 1.0
    pmo_score = max(-100.0, min(100.0, cur_pmo / pmo_norm_val * 100.0))

    # Component 2: histogram direction
    sig_window = sig_vals[-50:] if len(sig_vals) >= 50 else sig_vals
    hist_vals = [p - s for p, s in zip(pmo_vals[-len(sig_vals):], sig_vals)]
    hist_max = max(abs(v) for v in hist_vals) if hist_vals else 1.0
    hist_score = max(-100.0, min(100.0, hist / hist_max * 100.0)) if hist_max > 1e-9 else 0.0

    score = pmo_score * 0.6 + hist_score * 0.4
    score = max(-100.0, min(100.0, score))

    if score >= 55:
        signal = "strong_bull"
    elif score >= 15:
        signal = "bull"
    elif score <= -55:
        signal = "strong_bear"
    elif score <= -15:
        signal = "bear"
    else:
        signal = "neutral"

    # History series
    hist_pmo = pmo_vals[-history:]
    hist_sig = sig_vals[-history:] if len(sig_vals) >= history else sig_vals

    verdict = (
        f"PMO ({symbol}): {signal.replace('_', ' ')}. "
        f"PMO={cur_pmo:+.3f}, Signal={cur_sig:+.3f}, Hist={hist:+.3f}. "
        f"Rank={pmo_rank:.0f}th pct."
    )
    if cross_up:
        verdict += " Bullish cross."
    if cross_down:
        verdict += " Bearish cross."
    if overbought:
        verdict += " Overbought."
    if oversold:
        verdict += " Oversold."

    return PMOReport(
        symbol=symbol,
        pmo=round(cur_pmo, 4),
        pmo_signal=round(cur_sig, 4),
        pmo_histogram=round(hist, 4),
        pmo_bullish=cur_pmo > 0,
        above_signal=cur_pmo > cur_sig,
        cross_up=cross_up,
        cross_down=cross_down,
        pmo_pct_rank=round(pmo_rank, 1),
        overbought=overbought,
        oversold=oversold,
        score=round(score, 2),
        signal=signal,
        pmo_series=[round(v, 4) for v in hist_pmo],
        signal_series=[round(v, 4) for v in hist_sig],
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
