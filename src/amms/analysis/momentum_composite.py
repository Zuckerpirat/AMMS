"""Momentum composite score.

Aggregates multiple momentum indicators into a single composite score
to reduce noise from any single indicator.

Indicators used (equal weight by default):
  1. RSI (14): normalized to -50..+50 (RSI-50 mapped to ±50 range)
  2. ROC (20): normalized by capping at ±10%
  3. MACD histogram: normalized by comparing to recent histogram range
  4. Williams %R (14): normalized to -50..+50 (%R+50 maps to ±50 range)

Final score: -100 (extreme bearish) to +100 (extreme bullish)
  > +60: strong bullish
  +20 to +60: bullish
  -20 to +20: neutral
  -20 to -60: bearish
  < -60: strong bearish
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumComposite:
    symbol: str
    score: float                  # -100 to +100
    signal: str                   # "strong_bull"|"bull"|"neutral"|"bear"|"strong_bear"
    rsi_component: float | None   # normalized contribution
    roc_component: float | None
    macd_component: float | None
    wr_component: float | None
    n_components: int             # how many indicators contributed
    current_price: float
    bars_used: int


def _rsi_norm(bars: list, period: int = 14) -> float | None:
    """Compute RSI and normalize to -50..+50."""
    closes = [b.close for b in bars]
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1):]
    gains = [max(window[i] - window[i - 1], 0) for i in range(1, len(window))]
    losses = [max(window[i - 1] - window[i], 0) for i in range(1, len(window))]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    return (rsi - 50) * 1.0  # -50..+50


def _roc_norm(bars: list, period: int = 20) -> float | None:
    """Compute ROC and normalize by capping at ±10%."""
    closes = [b.close for b in bars]
    if len(closes) < period + 1:
        return None
    roc = (closes[-1] / closes[-(period + 1)] - 1) * 100
    return max(-50.0, min(50.0, roc * 5.0))  # 10% move → ±50


def _macd_norm(bars: list, fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    """Compute MACD histogram and normalize to -50..+50."""
    closes = [b.close for b in bars]
    needed = slow + signal
    if len(closes) < needed:
        return None

    def ema(data: list[float], span: int) -> list[float]:
        k = 2 / (span + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return result

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    macd_signal = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, macd_signal)]

    if not hist:
        return None

    current_hist = hist[-1]
    # Normalize by recent histogram range
    recent = hist[-signal:] if len(hist) >= signal else hist
    max_abs = max(abs(v) for v in recent) if recent else 0
    if max_abs < 1e-10:
        return 0.0
    return max(-50.0, min(50.0, current_hist / max_abs * 50))


def _wr_norm(bars: list, period: int = 14) -> float | None:
    """Compute Williams %R and normalize to -50..+50."""
    if len(bars) < period:
        return None
    window = bars[-period:]
    highest_high = max(b.high for b in window)
    lowest_low = min(b.low for b in window)
    current_close = bars[-1].close
    if highest_high == lowest_low:
        return 0.0
    wr = (highest_high - current_close) / (highest_high - lowest_low) * -100
    return wr + 50  # -100..0 → -50..+50


def compute(bars: list) -> MomentumComposite | None:
    """Compute momentum composite score.

    bars: list[Bar] — needs at least 35 bars for all indicators
    Returns None if insufficient data.
    """
    if len(bars) < 35:
        return None

    symbol = bars[0].symbol
    current_price = bars[-1].close

    rsi_c = _rsi_norm(bars)
    roc_c = _roc_norm(bars)
    macd_c = _macd_norm(bars)
    wr_c = _wr_norm(bars)

    components = [c for c in [rsi_c, roc_c, macd_c, wr_c] if c is not None]
    if not components:
        return None

    raw_score = sum(components) / len(components)
    # Scale from -50..+50 to -100..+100
    score = raw_score * 2.0
    score = max(-100.0, min(100.0, score))

    if score > 60:
        signal = "strong_bull"
    elif score > 20:
        signal = "bull"
    elif score > -20:
        signal = "neutral"
    elif score > -60:
        signal = "bear"
    else:
        signal = "strong_bear"

    return MomentumComposite(
        symbol=symbol,
        score=round(score, 1),
        signal=signal,
        rsi_component=round(rsi_c, 2) if rsi_c is not None else None,
        roc_component=round(roc_c, 2) if roc_c is not None else None,
        macd_component=round(macd_c, 2) if macd_c is not None else None,
        wr_component=round(wr_c, 2) if wr_c is not None else None,
        n_components=len(components),
        current_price=round(current_price, 4),
        bars_used=len(bars),
    )
