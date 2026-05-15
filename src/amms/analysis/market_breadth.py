"""Portfolio breadth analysis.

Measures the internal health of a portfolio by checking how many
positions are in bullish vs bearish technical states across multiple indicators.

Breadth metrics:
  - Pct above VWAP: how many positions trade above their VWAP
  - Pct in RSI uptrend (RSI > 50): how many have positive RSI momentum
  - Pct above SMA-20: how many trade above their 20-day moving average
  - Pct with positive OBV trend: volume flow supporting the move
  - Net breadth score: combined bullish% - bearish%

A strong portfolio has high breadth across all indicators.
Low breadth while price is up = divergence risk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BreadthStats:
    n_positions: int
    pct_above_vwap: float        # 0..100
    pct_rsi_above_50: float      # 0..100
    pct_above_sma20: float       # 0..100
    pct_obv_rising: float        # 0..100
    overall_score: float         # 0..100 — average of all four
    verdict: str                 # "strong" | "moderate" | "weak" | "deteriorating"
    detail: list[str]            # per-symbol status lines


def analyze_breadth(broker, data) -> BreadthStats | None:
    """Analyze portfolio breadth across open positions.

    broker: must support get_positions()
    data: must support get_bars(symbol, limit=N)
    Returns None if no open positions or no data available.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if not positions:
        return None

    from amms.features.vwap import vwap as compute_vwap
    from amms.features.momentum import rsi as compute_rsi

    above_vwap = 0
    rsi_above_50 = 0
    above_sma20 = 0
    obv_rising = 0
    total = 0
    detail: list[str] = []

    for pos in positions:
        sym = pos.symbol
        try:
            bars = data.get_bars(sym, limit=30)
        except Exception:
            continue
        if len(bars) < 5:
            continue

        total += 1
        price = bars[-1].close
        flags: list[str] = []

        # VWAP check
        vwap_val = compute_vwap(bars, 20)
        if vwap_val is not None:
            if price > vwap_val:
                above_vwap += 1
                flags.append("↑VWAP")
            else:
                flags.append("↓VWAP")

        # RSI check
        rsi_val = compute_rsi(bars, 14)
        if rsi_val is not None:
            if rsi_val > 50:
                rsi_above_50 += 1
                flags.append(f"RSI {rsi_val:.0f}↑")
            else:
                flags.append(f"RSI {rsi_val:.0f}↓")

        # SMA-20 check
        if len(bars) >= 20:
            sma20 = sum(b.close for b in bars[-20:]) / 20
            if price > sma20:
                above_sma20 += 1
                flags.append("↑SMA20")
            else:
                flags.append("↓SMA20")

        # OBV trend check (simplified: last 10 bars)
        try:
            from amms.features.obv import obv as compute_obv
            obv_result = compute_obv(bars)
            if obv_result is not None and obv_result.trend == "rising":
                obv_rising += 1
                flags.append("OBV↑")
            else:
                flags.append("OBV↓")
        except Exception:
            pass

        detail.append(f"  {sym:<6} ${price:.2f}  {' '.join(flags)}")

    if total == 0:
        return None

    pct_vwap = above_vwap / total * 100
    pct_rsi = rsi_above_50 / total * 100
    pct_sma = above_sma20 / total * 100
    pct_obv = obv_rising / total * 100

    overall = (pct_vwap + pct_rsi + pct_sma + pct_obv) / 4

    if overall >= 70:
        verdict = "strong"
    elif overall >= 50:
        verdict = "moderate"
    elif overall >= 30:
        verdict = "weak"
    else:
        verdict = "deteriorating"

    return BreadthStats(
        n_positions=total,
        pct_above_vwap=round(pct_vwap, 1),
        pct_rsi_above_50=round(pct_rsi, 1),
        pct_above_sma20=round(pct_sma, 1),
        pct_obv_rising=round(pct_obv, 1),
        overall_score=round(overall, 1),
        verdict=verdict,
        detail=detail,
    )
