"""Signal confluence analyzer.

Aggregates multiple technical indicators into a single directional score.
Helps identify when multiple signals align (confluence = higher confidence).

Indicators scored:
  RSI     : oversold (<30) = +1, overbought (>70) = -1
  MACD    : histogram > 0 = +0.5, < 0 = -0.5; signal crossover = ±1
  Bollinger: %B < 0.2 = +1 (near lower band), %B > 0.8 = -1 (near upper)
  ADX     : strong trend present = boosts absolute score by 0.5
  Stoch   : oversold (<20) = +1, overbought (>80) = -1; cross = ±1
  Z-score : z < -2 = +1, z > +2 = -1
  Momentum: 20d return positive = +0.5, negative = -0.5

Total score: positive = bullish, negative = bearish.
Normalized to -1..+1 range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from amms.data.bars import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfluenceSignal:
    symbol: str
    score: float          # -1..+1 normalized
    raw_score: float      # unnormalized sum
    max_possible: float   # theoretical max (depends on available signals)
    signals_checked: int
    bullish_signals: list[str]
    bearish_signals: list[str]
    verdict: str          # "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
    confidence: float     # 0..1


def analyze(bars: list[Bar]) -> ConfluenceSignal:
    """Run all available indicators and return a confluence score.

    Score interpretation:
      > 0.6  : strong bullish confluence
      > 0.3  : mild bullish
      -0.3..0.3 : neutral / mixed
      < -0.3 : mild bearish
      < -0.6 : strong bearish
    """
    symbol = bars[0].symbol if bars else "?"
    bullish: list[str] = []
    bearish: list[str] = []
    raw = 0.0
    max_pts = 0.0

    # --- RSI ---
    try:
        from amms.features.momentum import rsi as compute_rsi
        rsi_val = compute_rsi(bars, 14)
        if rsi_val is not None:
            max_pts += 1.0
            if rsi_val < 30:
                raw += 1.0
                bullish.append(f"RSI {rsi_val:.1f} (oversold)")
            elif rsi_val > 70:
                raw -= 1.0
                bearish.append(f"RSI {rsi_val:.1f} (overbought)")
    except Exception:
        pass

    # --- MACD ---
    try:
        from amms.features.momentum import macd as compute_macd
        macd_result = compute_macd(bars)
        if macd_result is not None:
            macd_line, signal_line, histogram = macd_result
            max_pts += 1.5
            if histogram > 0:
                raw += 0.5
                bullish.append(f"MACD hist +{histogram:.3f}")
            elif histogram < 0:
                raw -= 0.5
                bearish.append(f"MACD hist {histogram:.3f}")
            if macd_line > signal_line:
                raw += 0.5
                bullish.append("MACD above signal")
            else:
                raw -= 0.5
                bearish.append("MACD below signal")
    except Exception:
        pass

    # --- Bollinger %B ---
    try:
        from amms.features.bollinger import bollinger
        bb = bollinger(bars, 20)
        if bb is not None:
            max_pts += 1.0
            if bb.pct_b < 0.2:
                raw += 1.0
                bullish.append(f"BB %B {bb.pct_b:.2f} (near lower band)")
            elif bb.pct_b > 0.8:
                raw -= 1.0
                bearish.append(f"BB %B {bb.pct_b:.2f} (near upper band)")
    except Exception:
        pass

    # --- ADX (trend boost) ---
    try:
        from amms.features.adx import adx as compute_adx
        adx_result = compute_adx(bars, 14)
        if adx_result is not None and adx_result.trend_strength in ("strong", "very_strong", "extreme"):
            max_pts += 0.5
            if adx_result.direction == "bullish":
                raw += 0.5
                bullish.append(f"ADX {adx_result.adx:.1f} strong bullish trend")
            elif adx_result.direction == "bearish":
                raw -= 0.5
                bearish.append(f"ADX {adx_result.adx:.1f} strong bearish trend")
    except Exception:
        pass

    # --- Stochastic ---
    try:
        from amms.features.stochastic import stochastic
        stoch = stochastic(bars, 14, 3)
        if stoch is not None:
            max_pts += 1.0
            if stoch.zone == "oversold":
                raw += 1.0
                bullish.append(f"Stoch %K {stoch.k:.1f} (oversold)")
            elif stoch.zone == "overbought":
                raw -= 1.0
                bearish.append(f"Stoch %K {stoch.k:.1f} (overbought)")
            if stoch.signal == "bullish_cross":
                raw += 0.5
                bullish.append("Stochastic bullish cross")
            elif stoch.signal == "bearish_cross":
                raw -= 0.5
                bearish.append("Stochastic bearish cross")
    except Exception:
        pass

    # --- Z-score ---
    try:
        from amms.features.zscore import zscore
        z = zscore(bars, 20)
        if z is not None:
            max_pts += 1.0
            if z < -2.0:
                raw += 1.0
                bullish.append(f"Z-score {z:+.2f} (far below mean)")
            elif z < -1.0:
                raw += 0.5
                bullish.append(f"Z-score {z:+.2f} (below mean)")
            elif z > 2.0:
                raw -= 1.0
                bearish.append(f"Z-score {z:+.2f} (far above mean)")
            elif z > 1.0:
                raw -= 0.5
                bearish.append(f"Z-score {z:+.2f} (above mean)")
    except Exception:
        pass

    # --- Momentum (20d return) ---
    try:
        from amms.features.momentum import n_day_return
        mom = n_day_return(bars, 20)
        if mom is not None:
            max_pts += 0.5
            if mom > 0:
                raw += 0.5
                bullish.append(f"20d return +{mom * 100:.1f}%")
            else:
                raw -= 0.5
                bearish.append(f"20d return {mom * 100:.1f}%")
    except Exception:
        pass

    signals_checked = len(bullish) + len(bearish)
    if max_pts == 0:
        score = 0.0
        confidence = 0.0
    else:
        score = raw / max_pts
        score = max(-1.0, min(1.0, score))
        confidence = min(signals_checked / 5.0, 1.0)

    if score > 0.6:
        verdict = "strong_buy"
    elif score > 0.3:
        verdict = "buy"
    elif score < -0.6:
        verdict = "strong_sell"
    elif score < -0.3:
        verdict = "sell"
    else:
        verdict = "neutral"

    return ConfluenceSignal(
        symbol=symbol,
        score=round(score, 3),
        raw_score=round(raw, 2),
        max_possible=round(max_pts, 2),
        signals_checked=signals_checked,
        bullish_signals=bullish,
        bearish_signals=bearish,
        verdict=verdict,
        confidence=round(confidence, 2),
    )
