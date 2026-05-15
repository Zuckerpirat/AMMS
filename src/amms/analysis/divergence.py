"""RSI / price divergence detector.

Divergence occurs when price and momentum point in opposite directions:

  Bullish divergence: price makes a lower low, RSI makes a higher low.
    — suggests downtrend is losing momentum (potential reversal up).
  Bearish divergence: price makes a higher high, RSI makes a lower high.
    — suggests uptrend is losing momentum (potential reversal down).

Hidden divergence (continuation):
  Hidden bullish: price higher low, RSI lower low  → trend continues up.
  Hidden bearish: price lower high, RSI higher high → trend continues down.

The detector uses a configurable lookback window to identify swing pivots
and then compares the direction of price vs RSI at those pivots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amms.data.bars import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DivergenceResult:
    symbol: str
    divergence_type: str   # "bullish" | "bearish" | "hidden_bullish" | "hidden_bearish" | "none"
    confidence: float      # 0..1
    price_swing: str       # e.g. "lower low" or "higher high"
    rsi_swing: str         # e.g. "higher low" or "lower high"
    bars_checked: int
    reason: str


def _compute_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Compute RSI series from a list of closing prices."""
    if len(closes) < period + 1:
        return []
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi_values: list[float] = []
    for i in range(period, len(gains)):
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return rsi_values


def _find_pivots(values: list[float], window: int = 3) -> list[tuple[int, float]]:
    """Return list of (index, value) for local peaks and troughs.

    A pivot low is a point lower than all neighbors within `window`.
    Returns all pivots (both highs and lows).
    """
    pivots: list[tuple[int, float]] = []
    for i in range(window, len(values) - window):
        segment = values[i - window: i + window + 1]
        v = values[i]
        if v == min(segment) or v == max(segment):
            pivots.append((i, v))
    return pivots


def detect_divergence(
    bars: list[Bar],
    *,
    rsi_period: int = 14,
    lookback: int = 50,
    pivot_window: int = 3,
) -> DivergenceResult:
    """Detect RSI/price divergence for the most recent bars.

    Compares the last two significant swing pivots in both price and RSI.
    Returns the most recent divergence signal found.
    """
    symbol = bars[0].symbol if bars else "?"
    min_bars = rsi_period + lookback + pivot_window * 2 + 5

    if len(bars) < rsi_period + pivot_window * 2 + 5:
        return DivergenceResult(
            symbol=symbol,
            divergence_type="none",
            confidence=0.0,
            price_swing="n/a",
            rsi_swing="n/a",
            bars_checked=len(bars),
            reason=f"Insufficient bars ({len(bars)} < {rsi_period + pivot_window * 2 + 5} needed).",
        )

    recent_bars = bars[-lookback:] if len(bars) >= lookback else bars
    closes = [b.close for b in recent_bars]

    rsi_series = _compute_rsi(closes, rsi_period)
    if len(rsi_series) < pivot_window * 2 + 2:
        return DivergenceResult(
            symbol=symbol,
            divergence_type="none",
            confidence=0.0,
            price_swing="n/a",
            rsi_swing="n/a",
            bars_checked=len(recent_bars),
            reason="Insufficient RSI data for pivot detection.",
        )

    # Align closes with RSI (RSI starts at index rsi_period)
    aligned_closes = closes[rsi_period:]

    price_pivots = _find_pivots(aligned_closes, pivot_window)
    rsi_pivots = _find_pivots(rsi_series, pivot_window)

    if len(price_pivots) < 2 or len(rsi_pivots) < 2:
        return DivergenceResult(
            symbol=symbol,
            divergence_type="none",
            confidence=0.0,
            price_swing="n/a",
            rsi_swing="n/a",
            bars_checked=len(recent_bars),
            reason="Not enough pivots found in the lookback window.",
        )

    # Use the last two pivots for each
    p1_price, p2_price = price_pivots[-2][1], price_pivots[-1][1]
    p1_rsi, p2_rsi = rsi_pivots[-2][1], rsi_pivots[-1][1]

    price_dir = "higher" if p2_price > p1_price else "lower"
    rsi_dir = "higher" if p2_rsi > p1_rsi else "lower"

    # Determine pivot type (high or low) from last pivot
    last_price_idx = price_pivots[-1][0]
    price_segment = aligned_closes[max(0, last_price_idx - pivot_window): last_price_idx + pivot_window + 1]
    last_pivot_is_high = p2_price == max(price_segment) if price_segment else False

    if last_pivot_is_high:
        pivot_label = "high"
        price_swing = f"{price_dir} high"
        rsi_swing = f"{rsi_dir} high"
    else:
        pivot_label = "low"
        price_swing = f"{price_dir} low"
        rsi_swing = f"{rsi_dir} low"

    div_type = "none"
    confidence = 0.0
    reason = "No divergence detected."

    if pivot_label == "low":
        if price_dir == "lower" and rsi_dir == "higher":
            div_type = "bullish"
            confidence = _divergence_confidence(p1_price, p2_price, p1_rsi, p2_rsi)
            reason = "Price lower low + RSI higher low: bullish divergence (momentum building)."
        elif price_dir == "higher" and rsi_dir == "lower":
            div_type = "hidden_bullish"
            confidence = _divergence_confidence(p1_price, p2_price, p1_rsi, p2_rsi)
            reason = "Price higher low + RSI lower low: hidden bullish (uptrend continuation)."
    else:  # high pivot
        if price_dir == "higher" and rsi_dir == "lower":
            div_type = "bearish"
            confidence = _divergence_confidence(p1_price, p2_price, p1_rsi, p2_rsi)
            reason = "Price higher high + RSI lower high: bearish divergence (momentum fading)."
        elif price_dir == "lower" and rsi_dir == "higher":
            div_type = "hidden_bearish"
            confidence = _divergence_confidence(p1_price, p2_price, p1_rsi, p2_rsi)
            reason = "Price lower high + RSI higher high: hidden bearish (downtrend continuation)."

    return DivergenceResult(
        symbol=symbol,
        divergence_type=div_type,
        confidence=round(confidence, 2),
        price_swing=price_swing,
        rsi_swing=rsi_swing,
        bars_checked=len(recent_bars),
        reason=reason,
    )


def _divergence_confidence(p1_price: float, p2_price: float, p1_rsi: float, p2_rsi: float) -> float:
    """Estimate confidence from the magnitude of divergence.

    Larger price move + larger RSI divergence = higher confidence.
    Returns 0..1.
    """
    if p1_price == 0:
        return 0.5
    price_move_pct = abs(p2_price - p1_price) / p1_price * 100
    rsi_move = abs(p2_rsi - p1_rsi)
    # Normalize: 5% price move = good, 10 RSI points = good
    price_score = min(price_move_pct / 5.0, 1.0)
    rsi_score = min(rsi_move / 10.0, 1.0)
    return (price_score + rsi_score) / 2.0
