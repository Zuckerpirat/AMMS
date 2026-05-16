"""Detrended Price Oscillator (DPO).

The DPO removes trend from price to expose underlying cycles. Unlike most
oscillators, it does NOT use the most recent bars but compares price to a
displaced SMA to isolate cyclical behaviour.

Calculation:
  DPO = Close[t] - SMA(Close, period)[t - (period/2 + 1)]
  (i.e., today's price minus the SMA from (period/2 + 1) bars ago)

This eliminates longer-term trends and shows shorter oscillation cycles.

Interpretation:
  DPO > 0  → price above the mid-cycle SMA (positive phase)
  DPO < 0  → price below the mid-cycle SMA (negative phase)
  Peaks/troughs → turning points in the cycle
  Cycle length estimator: average distance between DPO peaks

Also provides:
  - Rolling DPO series for visual cycle analysis
  - Normalised score vs recent history
  - Overbought/oversold zones (DPO percentile rank)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DPOReport:
    symbol: str
    period: int
    displacement: int   # period // 2 + 1

    # Current DPO
    dpo: float              # current DPO value
    dpo_pct_rank: float     # 0-100 percentile vs recent history
    dpo_positive: bool

    # Cycle analysis from recent DPO series
    estimated_cycle: int | None   # bars between recent peaks (None if unclear)
    recent_peak: float | None     # highest DPO in recent window
    recent_trough: float | None   # lowest DPO in recent window

    # Signal
    score: float            # -100 to +100
    signal: str             # "overbought", "high", "neutral", "low", "oversold"

    # History series
    dpo_series: list[float]

    current_price: float
    bars_used: int
    verdict: str


def _sma(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    return [sum(series[i:i + period]) / period for i in range(len(series) - period + 1)]


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 20,
    history: int = 40,
) -> DPOReport | None:
    """Compute the Detrended Price Oscillator.

    bars: bar objects with .close attribute.
    period: SMA period. Displacement = period // 2 + 1.
    """
    displacement = period // 2 + 1
    min_bars = period + history + displacement + 5
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

    # Compute SMA series
    sma_vals = _sma(closes, period)
    # sma_vals[i] = SMA of closes[i .. i+period-1]
    # We need the SMA from `displacement` bars ago:
    # At index t (in closes), displaced SMA = sma_vals[t - period + 1 - displacement]
    # DPO[t] = closes[t] - sma[corresponding displaced position]

    # Build DPO series over the available range
    dpo_full = []
    # Start at the first t where both closes[t] and the displaced sma exist
    # sma_vals[j] corresponds to SMA ending at closes[j + period - 1]
    # We want SMA ending at t - displacement: j + period - 1 = t - displacement → j = t - displacement - period + 1
    # Need j >= 0 → t >= displacement + period - 1
    start_t = displacement + period - 1

    for t in range(start_t, n):
        j = t - displacement - period + 1
        if j < 0 or j >= len(sma_vals):
            continue
        dpo_full.append(closes[t] - sma_vals[j])

    if len(dpo_full) < 5:
        return None

    cur_dpo = dpo_full[-1]

    # Percentile rank
    dpo_window = dpo_full[-history:] if len(dpo_full) >= history else dpo_full
    pct_rank = sum(1 for v in dpo_window if v <= cur_dpo) / len(dpo_window) * 100.0

    # Cycle analysis: find peaks and troughs in dpo_window
    peaks = []
    troughs = []
    for i in range(1, len(dpo_window) - 1):
        if dpo_window[i] > dpo_window[i - 1] and dpo_window[i] > dpo_window[i + 1]:
            peaks.append((i, dpo_window[i]))
        elif dpo_window[i] < dpo_window[i - 1] and dpo_window[i] < dpo_window[i + 1]:
            troughs.append((i, dpo_window[i]))

    recent_peak   = max((v for _, v in peaks), default=None)
    recent_trough = min((v for _, v in troughs), default=None)

    # Estimated cycle length from average peak-to-peak distance
    est_cycle = None
    if len(peaks) >= 2:
        gaps = [peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)]
        est_cycle = round(sum(gaps) / len(gaps))

    # Score: map pct_rank to -100..+100 (extreme high = overbought = bearish)
    score = (pct_rank - 50) * 2.0  # 100th pct → +100, 0th → -100
    score = max(-100.0, min(100.0, score))

    # Signal (DPO oscillator labels differ: high pct = overbought cycle)
    if pct_rank >= 80:
        signal = "overbought"
    elif pct_rank >= 60:
        signal = "high"
    elif pct_rank <= 20:
        signal = "oversold"
    elif pct_rank <= 40:
        signal = "low"
    else:
        signal = "neutral"

    # History
    hist_series = dpo_full[-history:]

    verdict = (
        f"DPO ({symbol}, {period}): {signal}. "
        f"DPO={cur_dpo:+.3f}, rank={pct_rank:.0f}th pct."
    )
    if est_cycle is not None:
        verdict += f" Est. cycle ~{est_cycle} bars."

    return DPOReport(
        symbol=symbol,
        period=period,
        displacement=displacement,
        dpo=round(cur_dpo, 4),
        dpo_pct_rank=round(pct_rank, 1),
        dpo_positive=cur_dpo > 0,
        estimated_cycle=est_cycle,
        recent_peak=round(recent_peak, 4) if recent_peak is not None else None,
        recent_trough=round(recent_trough, 4) if recent_trough is not None else None,
        score=round(score, 2),
        signal=signal,
        dpo_series=[round(v, 4) for v in hist_series],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
