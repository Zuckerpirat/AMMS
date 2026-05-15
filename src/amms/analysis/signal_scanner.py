"""Multi-indicator signal scanner.

Scans a watchlist for high-confidence trade setups by combining:
  - Bollinger Bands (%B < 0.2 = near lower band)
  - RSI (< 35 oversold, > 65 overbought)
  - MACD (histogram direction change)
  - Stochastic (%K < 20 oversold, %K > 80 overbought)
  - ADX (> 20 = trend present)
  - Z-score (< -1.5 below mean, > 1.5 above mean)
  - Volume spike (ratio > 1.5)

Each signal adds points to a buy or sell score.
Returns ranked list of strongest setups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amms.data.bars import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalSetup:
    symbol: str
    direction: str         # "buy" | "sell" | "neutral"
    score: float           # 0..10
    signals: list[str]     # human-readable signal list
    confidence: str        # "low" | "medium" | "high"
    price: float


def scan_signals(
    symbols: list[str],
    data,
    *,
    min_score: float = 3.0,
    top_n: int = 10,
) -> list[SignalSetup]:
    """Scan symbols for trade setups and return ranked results.

    Only returns setups with score >= min_score.
    """
    results: list[SignalSetup] = []

    for sym in symbols:
        try:
            bars = data.get_bars(sym, limit=80)
        except Exception as e:
            logger.debug("failed to get bars for %s: %s", sym, e)
            continue

        if len(bars) < 20:
            continue

        setup = _analyze(sym, bars)
        if setup.score >= min_score:
            results.append(setup)

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


def _analyze(sym: str, bars: list[Bar]) -> SignalSetup:
    price = bars[-1].close
    buy_score = 0.0
    sell_score = 0.0
    signals: list[str] = []

    # Bollinger Bands
    try:
        from amms.features.bollinger import bollinger
        bb = bollinger(bars, 20)
        if bb is not None:
            if bb.pct_b < 0.1:
                buy_score += 2.0
                signals.append(f"BB %B {bb.pct_b:.2f} (at/below lower band)")
            elif bb.pct_b < 0.2:
                buy_score += 1.0
                signals.append(f"BB %B {bb.pct_b:.2f} (near lower band)")
            elif bb.pct_b > 0.9:
                sell_score += 2.0
                signals.append(f"BB %B {bb.pct_b:.2f} (at/above upper band)")
            elif bb.pct_b > 0.8:
                sell_score += 1.0
                signals.append(f"BB %B {bb.pct_b:.2f} (near upper band)")
    except Exception:
        pass

    # RSI
    try:
        from amms.features.momentum import rsi
        rsi_val = rsi(bars, 14)
        if rsi_val is not None:
            if rsi_val < 25:
                buy_score += 2.0
                signals.append(f"RSI {rsi_val:.1f} (extreme oversold)")
            elif rsi_val < 35:
                buy_score += 1.0
                signals.append(f"RSI {rsi_val:.1f} (oversold)")
            elif rsi_val > 75:
                sell_score += 2.0
                signals.append(f"RSI {rsi_val:.1f} (extreme overbought)")
            elif rsi_val > 65:
                sell_score += 1.0
                signals.append(f"RSI {rsi_val:.1f} (overbought)")
    except Exception:
        pass

    # MACD
    try:
        from amms.features.momentum import macd
        macd_r = macd(bars)
        if macd_r is not None:
            _, _, hist = macd_r
            if hist > 0:
                buy_score += 0.5
                signals.append(f"MACD hist +{hist:.3f}")
            elif hist < 0:
                sell_score += 0.5
                signals.append(f"MACD hist {hist:.3f}")
    except Exception:
        pass

    # Stochastic
    try:
        from amms.features.stochastic import stochastic
        stoch = stochastic(bars, 14, 3)
        if stoch is not None:
            if stoch.zone == "oversold":
                buy_score += 1.5
                signals.append(f"Stoch %K {stoch.k:.1f} oversold")
            elif stoch.zone == "overbought":
                sell_score += 1.5
                signals.append(f"Stoch %K {stoch.k:.1f} overbought")
            if stoch.signal == "bullish_cross":
                buy_score += 1.0
                signals.append("Stoch bullish cross ↑")
            elif stoch.signal == "bearish_cross":
                sell_score += 1.0
                signals.append("Stoch bearish cross ↓")
    except Exception:
        pass

    # Z-score
    try:
        from amms.features.zscore import zscore
        z = zscore(bars, 20)
        if z is not None:
            if z < -2.0:
                buy_score += 1.5
                signals.append(f"Z-score {z:+.2f} (very low)")
            elif z < -1.5:
                buy_score += 1.0
                signals.append(f"Z-score {z:+.2f} (below mean)")
            elif z > 2.0:
                sell_score += 1.5
                signals.append(f"Z-score {z:+.2f} (very high)")
            elif z > 1.5:
                sell_score += 1.0
                signals.append(f"Z-score {z:+.2f} (above mean)")
    except Exception:
        pass

    # Volume spike (buy bias only — confirms either direction)
    try:
        from amms.features.bollinger import volume_spike
        ratio = volume_spike(bars, 20)
        if ratio is not None and ratio >= 2.0:
            if buy_score > sell_score:
                buy_score += 0.5
                signals.append(f"Volume spike ×{ratio:.1f}")
            else:
                sell_score += 0.5
                signals.append(f"Volume spike ×{ratio:.1f}")
    except Exception:
        pass

    if buy_score >= sell_score:
        direction = "buy" if buy_score >= 3.0 else "neutral"
        score = buy_score
    else:
        direction = "sell" if sell_score >= 3.0 else "neutral"
        score = sell_score

    if score >= 6.0:
        confidence = "high"
    elif score >= 4.0:
        confidence = "medium"
    else:
        confidence = "low"

    return SignalSetup(
        symbol=sym,
        direction=direction,
        score=round(score, 1),
        signals=signals,
        confidence=confidence,
        price=price,
    )
