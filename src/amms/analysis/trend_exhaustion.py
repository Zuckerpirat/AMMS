"""Trend Exhaustion Detector.

Detects when an ongoing trend shows signs of exhaustion — the price
has moved far but momentum is fading. Uses multiple signals:

  1. Price extension: how far above/below SMA-20 and SMA-50
  2. RSI divergence: price making new highs while RSI makes lower highs
     (or new lows while RSI makes higher lows)
  3. Momentum decay: 5-bar ROC is smaller than 20-bar ROC (deceleration)
  4. ATR contraction: recent ATR shrinking on trend extension
  5. Bar rejection: large upper/lower wicks on last 3 bars

Each signal contributes to an exhaustion score 0-100.
Higher = more exhausted / more likely to reverse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExhaustionReport:
    symbol: str
    exhaustion_score: float   # 0-100; higher = more exhausted
    exhaustion_label: str     # "high", "moderate", "low", "none"
    trend_direction: str      # "up" or "down" (direction of current trend)

    # Component scores (each 0-1)
    price_extension: float    # how far from SMA (0=normal, 1=extreme)
    rsi_divergence: bool      # True if divergence detected
    momentum_decay: bool      # 5-bar ROC decelerating vs 20-bar
    atr_contraction: bool     # ATR shrinking on trend
    bar_rejection: float      # 0=no wicks, 1=strong wicks against trend

    # Indicator values
    current_price: float
    sma20: float | None
    sma50: float | None
    rsi: float
    roc5: float
    roc20: float
    atr_pct: float
    pct_above_sma20: float

    bars_used: int
    verdict: str


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


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


def _roc(closes: list[float], period: int) -> float:
    if len(closes) <= period or closes[-(period + 1)] <= 0:
        return 0.0
    return (closes[-1] / closes[-(period + 1)] - 1.0) * 100.0


def _atr_pct(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h = float(bars[-i].high)
        lo = float(bars[-i].low)
        pc = float(bars[-i - 1].close)
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    atr = sum(trs) / len(trs) if trs else 0.0
    last_close = float(bars[-1].close)
    return atr / last_close * 100.0 if last_close > 0 else 0.0


def _wick_score(bars: list, n: int = 3, trend: str = "up") -> float:
    """0-1: rejection wick score for the last n bars."""
    if len(bars) < n:
        return 0.0
    scores = []
    for bar in bars[-n:]:
        try:
            h = float(bar.high)
            lo = float(bar.low)
            c = float(bar.close)
            o = float(bar.open) if hasattr(bar, "open") else (h + lo) / 2
            body = abs(c - o)
            bar_range = h - lo
            if bar_range < 1e-9:
                scores.append(0.0)
                continue
            if trend == "up":
                upper_wick = h - max(c, o)
                scores.append(min(1.0, upper_wick / bar_range))
            else:
                lower_wick = min(c, o) - lo
                scores.append(min(1.0, lower_wick / bar_range))
        except Exception:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def analyze(bars: list, *, symbol: str = "") -> ExhaustionReport | None:
    """Detect trend exhaustion from bar history.

    bars: bar objects with .close, .high, .low attributes.
    Needs at least 55 bars for full computation.
    """
    if not bars or len(bars) < 55:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    rsi_val = _rsi(closes)
    roc5 = _roc(closes, 5)
    roc20 = _roc(closes, 20)
    atr = _atr_pct(bars)

    # Determine trend direction from SMA slope
    if sma20 is not None and len(closes) >= 25:
        sma20_old = sum(closes[-25:-5]) / 20.0
        trend_dir = "up" if sma20 > sma20_old else "down"
    else:
        trend_dir = "up" if roc20 >= 0 else "down"

    # 1. Price extension from SMA-20
    pct_from_sma20 = ((current - sma20) / sma20 * 100.0) if sma20 and sma20 > 0 else 0.0
    if trend_dir == "up":
        extension = min(1.0, max(0.0, pct_from_sma20 / 10.0))  # 10% above = score 1
    else:
        extension = min(1.0, max(0.0, -pct_from_sma20 / 10.0))

    # 2. RSI divergence: new price extreme but RSI not confirming
    rsi_high_threshold = 70.0 if trend_dir == "up" else 30.0
    if trend_dir == "up":
        rsi_div = rsi_val < 60.0 and current > (sma50 * 1.05 if sma50 else current)
    else:
        rsi_div = rsi_val > 40.0 and current < (sma50 * 0.95 if sma50 else current)

    # 3. Momentum decay: abs(roc5) < abs(roc20) * 0.5
    mom_decay = abs(roc5) < abs(roc20) * 0.5 and abs(roc20) > 1.0

    # 4. ATR contraction: compare recent ATR to older ATR
    atr_contraction = False
    if len(bars) >= 30:
        atr_old = _atr_pct(bars[:-14], period=14)
        atr_contraction = (atr < atr_old * 0.7 and atr_old > 0.5)

    # 5. Wick rejection
    wick = _wick_score(bars, n=3, trend=trend_dir)

    # Composite score
    component_scores = [
        extension * 30,
        25.0 if rsi_div else 0.0,
        20.0 if mom_decay else 0.0,
        10.0 if atr_contraction else 0.0,
        wick * 15,
    ]
    total = sum(component_scores)

    if total >= 65:
        label = "high"
    elif total >= 40:
        label = "moderate"
    elif total >= 20:
        label = "low"
    else:
        label = "none"

    # Verdict
    parts = [f"exhaustion {label} ({total:.0f}/100, {trend_dir}-trend)"]
    if extension > 0.5:
        parts.append(f"price extended {pct_from_sma20:+.1f}% from SMA-20")
    if rsi_div:
        parts.append(f"RSI divergence (RSI {rsi_val:.0f})")
    if mom_decay:
        parts.append("momentum decelerating")
    if wick > 0.4:
        parts.append(f"wick rejection {wick:.0f}")

    verdict = "Trend exhaustion: " + "; ".join(parts) + "."

    return ExhaustionReport(
        symbol=symbol,
        exhaustion_score=round(total, 1),
        exhaustion_label=label,
        trend_direction=trend_dir,
        price_extension=round(extension, 3),
        rsi_divergence=rsi_div,
        momentum_decay=mom_decay,
        atr_contraction=atr_contraction,
        bar_rejection=wick,
        current_price=round(current, 4),
        sma20=round(sma20, 4) if sma20 else None,
        sma50=round(sma50, 4) if sma50 else None,
        rsi=round(rsi_val, 1),
        roc5=round(roc5, 3),
        roc20=round(roc20, 3),
        atr_pct=round(atr, 3),
        pct_above_sma20=round(pct_from_sma20, 3),
        bars_used=len(bars),
        verdict=verdict,
    )
