"""Multi-indicator signal aggregator.

Combines multiple technical indicators into a single directional score
(-100 to +100) with individual signal votes and a consensus signal.

Indicators included (all computed from OHLCV bars):
  1. RSI-14: <30 bullish, >70 bearish, else neutral
  2. MACD: signal line crossover direction
  3. ROC-20: positive/negative momentum
  4. Williams %R-14: <-80 bullish (oversold), >-20 bearish (overbought)
  5. Price vs SMA-20: above=bull, below=bear
  6. Price vs SMA-50: above=bull, below=bear
  7. Stochastic %K (fast): <20 bull, >80 bear
  8. OBV trend: rising=bull, falling=bear

Each indicator votes +1 (bull), -1 (bear), or 0 (neutral).
Final score = (sum of votes / max_votes) × 100.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorVote:
    name: str
    value: float | None
    vote: int          # +1, 0, -1
    signal: str        # "bull" / "neutral" / "bear"


@dataclass(frozen=True)
class AggregatedSignal:
    symbol: str
    score: float               # -100 to +100
    signal: str                # "strong_bull" / "bull" / "neutral" / "bear" / "strong_bear"
    bull_votes: int
    bear_votes: int
    neutral_votes: int
    votes: list[IndicatorVote]
    current_price: float
    bars_used: int
    verdict: str


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))][-period:]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _macd(closes: list[float]) -> float | None:
    """Returns MACD histogram (MACD line - signal line)."""
    if len(closes) < 35:
        return None
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if len(ema12) < 9 or len(ema26) < 9:
        return None
    n = min(len(ema12), len(ema26))
    macd_line = [ema12[i + len(ema12) - n] - ema26[i + len(ema26) - n] for i in range(n)]
    signal_line = _ema(macd_line, 9)
    if not signal_line:
        return None
    return macd_line[-1] - signal_line[-1]


def _stochastic_k(bars: list, period: int = 14) -> float | None:
    """Fast %K."""
    if len(bars) < period:
        return None
    recent = bars[-period:]
    high = max(float(b.high) for b in recent)
    low = min(float(b.low) for b in recent)
    close = float(bars[-1].close)
    if high == low:
        return 50.0
    return (close - low) / (high - low) * 100


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _williams_r(bars: list, period: int = 14) -> float | None:
    if len(bars) < period:
        return None
    recent = bars[-period:]
    high = max(float(b.high) for b in recent)
    low = min(float(b.low) for b in recent)
    close = float(bars[-1].close)
    if high == low:
        return -50.0
    return (high - close) / (high - low) * -100


def _obv_trend(bars: list, period: int = 20) -> float | None:
    """OBV slope: positive = accumulating."""
    if len(bars) < period + 1:
        return None
    obvs = [0.0]
    for i in range(1, len(bars)):
        c = float(bars[i].close)
        pc = float(bars[i - 1].close)
        v = float(bars[i].volume) if hasattr(bars[i], "volume") else 0.0
        if c > pc:
            obvs.append(obvs[-1] + v)
        elif c < pc:
            obvs.append(obvs[-1] - v)
        else:
            obvs.append(obvs[-1])

    recent_obvs = obvs[-period:]
    n = len(recent_obvs)
    if n < 2:
        return None
    slope = (recent_obvs[-1] - recent_obvs[0]) / n
    return slope


def compute(bars: list, *, symbol: str = "") -> AggregatedSignal | None:
    """Aggregate multiple indicator votes into a directional signal.

    bars: list of bars with .high .low .close .volume
    symbol: ticker name for display

    Returns None if fewer than 35 bars.
    """
    if not bars or len(bars) < 35:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    current_price = closes[-1]
    votes: list[IndicatorVote] = []

    def add_vote(name: str, value: float | None, vote: int):
        signal = "bull" if vote > 0 else ("bear" if vote < 0 else "neutral")
        votes.append(IndicatorVote(name=name, value=round(value, 2) if value is not None else None, vote=vote, signal=signal))

    # RSI
    rsi_val = _rsi(closes)
    if rsi_val is not None:
        if rsi_val < 35:
            add_vote("RSI-14", rsi_val, +1)
        elif rsi_val > 65:
            add_vote("RSI-14", rsi_val, -1)
        else:
            add_vote("RSI-14", rsi_val, 0)

    # MACD histogram
    macd_hist = _macd(closes)
    if macd_hist is not None:
        add_vote("MACD hist", macd_hist, +1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0))

    # ROC-20
    if len(closes) >= 21:
        roc = (closes[-1] / closes[-21] - 1) * 100
        add_vote("ROC-20", roc, +1 if roc > 2 else (-1 if roc < -2 else 0))

    # Williams %R
    wr = _williams_r(bars)
    if wr is not None:
        if wr < -80:
            add_vote("Williams %%R", wr, +1)
        elif wr > -20:
            add_vote("Williams %%R", wr, -1)
        else:
            add_vote("Williams %%R", wr, 0)

    # SMA-20
    sma20 = _sma(closes, 20)
    if sma20 is not None:
        add_vote("SMA-20", sma20, +1 if current_price > sma20 else -1)

    # SMA-50
    sma50 = _sma(closes, 50)
    if sma50 is not None:
        add_vote("SMA-50", sma50, +1 if current_price > sma50 else -1)

    # Stochastic %K
    stoch = _stochastic_k(bars)
    if stoch is not None:
        if stoch < 20:
            add_vote("Stoch %%K", stoch, +1)
        elif stoch > 80:
            add_vote("Stoch %%K", stoch, -1)
        else:
            add_vote("Stoch %%K", stoch, 0)

    # OBV trend
    obv_slope = _obv_trend(bars)
    if obv_slope is not None:
        add_vote("OBV trend", obv_slope, +1 if obv_slope > 0 else (-1 if obv_slope < 0 else 0))

    if not votes:
        return None

    total_votes = len(votes)
    bull = sum(1 for v in votes if v.vote > 0)
    bear = sum(1 for v in votes if v.vote < 0)
    neutral = sum(1 for v in votes if v.vote == 0)

    net = bull - bear
    score = net / total_votes * 100

    if score >= 60:
        signal = "strong_bull"
    elif score >= 25:
        signal = "bull"
    elif score <= -60:
        signal = "strong_bear"
    elif score <= -25:
        signal = "bear"
    else:
        signal = "neutral"

    verdict = {
        "strong_bull": "Strong bullish consensus — most indicators aligned up",
        "bull": "Bullish bias — majority of indicators point up",
        "neutral": "Mixed signals — no clear directional consensus",
        "bear": "Bearish bias — majority of indicators point down",
        "strong_bear": "Strong bearish consensus — most indicators aligned down",
    }.get(signal, "No consensus")

    return AggregatedSignal(
        symbol=symbol,
        score=round(score, 1),
        signal=signal,
        bull_votes=bull,
        bear_votes=bear,
        neutral_votes=neutral,
        votes=votes,
        current_price=round(current_price, 2),
        bars_used=len(bars),
        verdict=verdict,
    )
