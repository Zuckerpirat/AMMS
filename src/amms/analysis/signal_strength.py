"""Technical Signal Strength Aggregator.

Combines multiple independent technical signals into a single
strength score for a symbol. Each signal contributes a vote
from -2 (strong bear) to +2 (strong bull).

Signals used:
  1. Trend (price vs SMA-50, SMA-200 cross) — weight 2
  2. Momentum (ROC-20) — weight 2
  3. RSI positioning — weight 1
  4. Bollinger Band position — weight 1
  5. Volume confirmation — weight 1
  6. EMA slope (5-period EMA angle) — weight 1

Total vote range: -8 to +8
Score: normalised to 0-100 (50=neutral, >65=bull, <35=bear)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalComponent:
    name: str
    vote: int        # -2 to +2
    weight: int
    weighted_vote: int  # vote × weight
    description: str


@dataclass(frozen=True)
class SignalStrengthReport:
    symbol: str
    score: float          # 0-100 (50=neutral)
    grade: str            # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    total_vote: int       # raw sum of weighted votes
    max_possible: int     # maximum possible vote
    components: list[SignalComponent]
    current_price: float
    bars_used: int
    verdict: str


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for v in closes[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    last = changes[-period:]
    gains = [max(0.0, c) for c in last]
    losses = [abs(min(0.0, c)) for c in last]
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def analyze(bars: list, *, symbol: str = "") -> SignalStrengthReport | None:
    """Aggregate multiple technical signals into a strength score.

    bars: bar objects with .close, .high, .low, .volume attributes.
    Requires at least 55 bars.
    """
    if not bars or len(bars) < 55:
        return None

    try:
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        vols = [float(b.volume) if hasattr(b, "volume") else 1.0 for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    components: list[SignalComponent] = []

    # 1. Trend: price vs SMA-50 and SMA-200 (weight 2)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    if sma50 is not None:
        if sma200 is not None:
            # Golden/death cross
            if current > sma50 > sma200:
                trend_vote = 2
                trend_desc = f"price>{sma50:.0f}>{sma200:.0f} (golden)"
            elif current < sma50 < sma200:
                trend_vote = -2
                trend_desc = f"price<{sma50:.0f}<{sma200:.0f} (death)"
            elif current > sma50:
                trend_vote = 1
                trend_desc = f"above SMA-50 ({(current/sma50-1)*100:+.1f}%)"
            else:
                trend_vote = -1
                trend_desc = f"below SMA-50 ({(current/sma50-1)*100:+.1f}%)"
        else:
            trend_vote = 1 if current > sma50 else -1
            trend_desc = f"{'above' if trend_vote > 0 else 'below'} SMA-50"
    else:
        trend_vote = 0
        trend_desc = "insufficient data"
    components.append(SignalComponent("Trend", trend_vote, 2, trend_vote * 2, trend_desc))

    # 2. Momentum ROC-20 (weight 2)
    if len(closes) >= 21 and closes[-21] > 0:
        roc20 = (current / closes[-21] - 1.0) * 100.0
        if roc20 > 5:
            mom_vote = 2
        elif roc20 > 1:
            mom_vote = 1
        elif roc20 < -5:
            mom_vote = -2
        elif roc20 < -1:
            mom_vote = -1
        else:
            mom_vote = 0
        mom_desc = f"ROC-20: {roc20:+.1f}%"
    else:
        mom_vote = 0
        roc20 = 0.0
        mom_desc = "insufficient data"
    components.append(SignalComponent("Momentum", mom_vote, 2, mom_vote * 2, mom_desc))

    # 3. RSI (weight 1)
    rsi_val = _rsi(closes)
    if rsi_val > 70:
        rsi_vote = -1  # overbought
        rsi_desc = f"RSI {rsi_val:.0f} overbought"
    elif rsi_val > 55:
        rsi_vote = 1
        rsi_desc = f"RSI {rsi_val:.0f} bullish"
    elif rsi_val < 30:
        rsi_vote = 1  # oversold = potential reversal
        rsi_desc = f"RSI {rsi_val:.0f} oversold (opportunity)"
    elif rsi_val < 45:
        rsi_vote = -1
        rsi_desc = f"RSI {rsi_val:.0f} bearish"
    else:
        rsi_vote = 0
        rsi_desc = f"RSI {rsi_val:.0f} neutral"
    components.append(SignalComponent("RSI", rsi_vote, 1, rsi_vote, rsi_desc))

    # 4. Bollinger Band position (weight 1)
    if len(closes) >= 20:
        sma20 = sum(closes[-20:]) / 20.0
        std20 = (sum((c - sma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        band_range = upper_bb - lower_bb
        pos = (current - lower_bb) / band_range if band_range > 0 else 0.5
        if pos > 0.8:
            bb_vote = -1
            bb_desc = f"BB position {pos:.0%} (near upper)"
        elif pos > 0.6:
            bb_vote = 1
            bb_desc = f"BB position {pos:.0%} (upper half)"
        elif pos < 0.2:
            bb_vote = 1  # oversold within BB
            bb_desc = f"BB position {pos:.0%} (near lower — opportunity)"
        elif pos < 0.4:
            bb_vote = -1
            bb_desc = f"BB position {pos:.0%} (lower half)"
        else:
            bb_vote = 0
            bb_desc = f"BB position {pos:.0%} (middle)"
    else:
        bb_vote = 0
        bb_desc = "insufficient data"
    components.append(SignalComponent("BB Position", bb_vote, 1, bb_vote, bb_desc))

    # 5. Volume confirmation (weight 1)
    if len(vols) >= 20:
        vol5 = sum(vols[-5:]) / 5.0
        vol20 = sum(vols[-20:]) / 20.0
        vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0
        # Rising volume on up-move = bullish; rising on down-move = bearish
        roc5 = (closes[-1] / closes[-6] - 1.0) * 100.0 if len(closes) >= 6 and closes[-6] > 0 else 0.0
        if vol_ratio > 1.2 and roc5 > 0:
            vol_vote = 1
            vol_desc = f"vol {vol_ratio:.1f}× on up-move"
        elif vol_ratio > 1.2 and roc5 < 0:
            vol_vote = -1
            vol_desc = f"vol {vol_ratio:.1f}× on down-move"
        else:
            vol_vote = 0
            vol_desc = f"vol ratio {vol_ratio:.1f}×"
    else:
        vol_vote = 0
        vol_desc = "insufficient data"
    components.append(SignalComponent("Volume", vol_vote, 1, vol_vote, vol_desc))

    # 6. EMA-5 slope (weight 1)
    ema5 = _ema(closes, 5)
    if ema5 is not None and len(closes) >= 10:
        ema5_old = _ema(closes[:-5], 5)
        if ema5_old and ema5_old > 0:
            ema_slope = (ema5 - ema5_old) / ema5_old * 100.0
            if ema_slope > 0.5:
                ema_vote = 1
                ema_desc = f"EMA-5 slope {ema_slope:+.2f}% (rising)"
            elif ema_slope < -0.5:
                ema_vote = -1
                ema_desc = f"EMA-5 slope {ema_slope:+.2f}% (falling)"
            else:
                ema_vote = 0
                ema_desc = f"EMA-5 slope {ema_slope:+.2f}% (flat)"
        else:
            ema_vote = 0
            ema_desc = "EMA slope N/A"
    else:
        ema_vote = 0
        ema_desc = "insufficient data"
    components.append(SignalComponent("EMA Slope", ema_vote, 1, ema_vote, ema_desc))

    # Aggregate
    total_vote = sum(c.weighted_vote for c in components)
    max_possible = sum(c.weight * 2 for c in components)
    score = (total_vote + max_possible) / (2 * max_possible) * 100.0

    if score >= 70:
        grade = "strong_bull"
    elif score >= 60:
        grade = "bull"
    elif score <= 30:
        grade = "strong_bear"
    elif score <= 40:
        grade = "bear"
    else:
        grade = "neutral"

    verdict_parts = [f"{grade} ({score:.0f}/100)"]
    strong = [c for c in components if abs(c.vote) == 2]
    if strong:
        verdict_parts.append("; ".join(f"{c.name}: {c.description}" for c in strong[:2]))

    verdict = "Signal strength: " + "; ".join(verdict_parts) + "."

    return SignalStrengthReport(
        symbol=symbol,
        score=round(score, 1),
        grade=grade,
        total_vote=total_vote,
        max_possible=max_possible,
        components=components,
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
