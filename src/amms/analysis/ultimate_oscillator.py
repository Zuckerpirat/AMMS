"""Ultimate Oscillator (UO) Analyser.

Larry Williams' multi-period oscillator using buying pressure relative
to true range across three timeframes to reduce false signals.

Algorithm:
  TrueHigh    = max(High, PrevClose)
  TrueLow     = min(Low,  PrevClose)
  TrueRange   = TrueHigh - TrueLow
  BuyPressure = Close - TrueLow

  Average(period) = sum(BP, period) / sum(TR, period)

  UO = 100 × (4×Avg7 + 2×Avg14 + 1×Avg28) / (4 + 2 + 1)

Signal zones (Williams' original):
  UO < 30  → oversold  → potential buy divergence setup
  UO > 70  → overbought → potential sell setup
  45-65    → neutral

Also tracks divergence: price making new lows while UO rising (bullish)
or price new highs while UO falling (bearish).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UOReport:
    symbol: str

    uo: float           # current Ultimate Oscillator (0-100)
    avg7: float         # 7-period raw average
    avg14: float        # 14-period raw average
    avg28: float        # 28-period raw average

    overbought: bool    # uo > 70
    oversold: bool      # uo < 30
    slope_up: bool      # uo rising vs 3 bars ago

    # Divergence
    bull_divergence: bool  # price lower low, UO higher low
    bear_divergence: bool  # price higher high, UO lower high

    score: float        # -100..+100
    signal: str         # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    history: list[float]
    bars_used: int
    verdict: str


def _rolling_sum(values: list[float], period: int) -> float:
    return sum(values[-period:]) if len(values) >= period else 0.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
    w1: float = 4.0,
    w2: float = 2.0,
    w3: float = 1.0,
    history: int = 20,
) -> UOReport | None:
    """Compute Ultimate Oscillator.

    bars: bar objects with .high, .low, .close attributes.
    period1 / period2 / period3: fast, medium, slow lookbacks.
    w1 / w2 / w3: weights for each period.
    """
    min_bars = period3 + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    if closes[-1] <= 0:
        return None

    n = len(closes)

    # Compute BP and TR series
    bp_list: list[float] = []
    tr_list: list[float] = []

    for i in range(1, n):
        true_high = max(highs[i], closes[i - 1])
        true_low  = min(lows[i],  closes[i - 1])
        bp_list.append(closes[i] - true_low)
        tr_list.append(max(true_high - true_low, 1e-9))

    if len(bp_list) < period3 + history:
        return None

    # UO series
    uo_series: list[float] = []
    for t in range(period3 - 1, len(bp_list)):
        end = t + 1
        a1_bp = sum(bp_list[max(0, end - period1): end])
        a1_tr = sum(tr_list[max(0, end - period1): end])
        a2_bp = sum(bp_list[max(0, end - period2): end])
        a2_tr = sum(tr_list[max(0, end - period2): end])
        a3_bp = sum(bp_list[max(0, end - period3): end])
        a3_tr = sum(tr_list[max(0, end - period3): end])
        avg1 = a1_bp / a1_tr if a1_tr > 1e-9 else 0.5
        avg2 = a2_bp / a2_tr if a2_tr > 1e-9 else 0.5
        avg3 = a3_bp / a3_tr if a3_tr > 1e-9 else 0.5
        uo = (w1 * avg1 + w2 * avg2 + w3 * avg3) / (w1 + w2 + w3) * 100.0
        uo_series.append(uo)

    if len(uo_series) < history + 4:
        return None

    uo_val  = uo_series[-1]
    uo_3ago = uo_series[-4] if len(uo_series) >= 4 else uo_series[0]

    # Component averages at current bar
    end = len(bp_list)
    avg7  = sum(bp_list[-period1:]) / max(sum(tr_list[-period1:]), 1e-9)
    avg14 = sum(bp_list[-period2:]) / max(sum(tr_list[-period2:]), 1e-9)
    avg28 = sum(bp_list[-period3:]) / max(sum(tr_list[-period3:]), 1e-9)

    overbought = uo_val > 70.0
    oversold   = uo_val < 30.0
    slope_up   = uo_val > uo_3ago

    # Divergence check over recent history
    recent_uo    = uo_series[-history:]
    recent_close = closes[-(history + 1):]  # one extra for alignment

    # Bullish divergence: price lower low, UO higher low
    half = len(recent_uo) // 2
    price_lower_low = min(recent_close[:half]) > min(recent_close[half:])  # recent lows lower
    uo_higher_low   = min(recent_uo[:half])   < min(recent_uo[half:])     # UO lows higher
    bull_div = price_lower_low and uo_higher_low

    price_higher_high = max(recent_close[:half]) < max(recent_close[half:])
    uo_lower_high     = max(recent_uo[:half])    > max(recent_uo[half:])
    bear_div = price_higher_high and uo_lower_high

    # Score
    pos_score = (uo_val - 50.0) * 2.0
    div_bonus = 20.0 if bull_div else (-20.0 if bear_div else 0.0)
    score = max(-100.0, min(100.0, pos_score + div_bonus))

    if oversold and (slope_up or bull_div):
        signal = "strong_buy"
    elif oversold or score <= -50:
        signal = "buy"
    elif overbought and (not slope_up or bear_div):
        signal = "strong_sell"
    elif overbought or score >= 50:
        signal = "sell"
    else:
        signal = "neutral"

    div_str = ""
    if bull_div:
        div_str = " Bullish divergence."
    elif bear_div:
        div_str = " Bearish divergence."

    ob_str = " [OVERBOUGHT]" if overbought else (" [OVERSOLD]" if oversold else "")
    verdict = (
        f"UO ({symbol}): {uo_val:.1f}{ob_str}. "
        f"Avg7={avg7:.3f}, Avg14={avg14:.3f}, Avg28={avg28:.3f}. "
        f"Signal: {signal.replace('_', ' ')}.{div_str}"
    )

    return UOReport(
        symbol=symbol,
        uo=round(uo_val, 2),
        avg7=round(avg7, 4),
        avg14=round(avg14, 4),
        avg28=round(avg28, 4),
        overbought=overbought,
        oversold=oversold,
        slope_up=slope_up,
        bull_divergence=bull_div,
        bear_divergence=bear_div,
        score=round(score, 2),
        signal=signal,
        history=[round(v, 2) for v in recent_uo],
        bars_used=n,
        verdict=verdict,
    )
