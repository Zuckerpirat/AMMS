"""Intraday momentum analysis.

Analyzes short-term price momentum within a trading session using
minute or hourly bars. Identifies:
  - Opening gap and first-hour direction
  - VWAP relationship (above/below as momentum signal)
  - Accumulation/distribution (price weighted by volume)
  - Momentum fade detection (opening move reversal)
  - Session high/low position (price location as % of range)

Designed for intraday bars (e.g. 1m, 5m, 15m, 1h).
Also works with daily bars as a trend confirmation tool.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntradayMomentum:
    symbol: str
    current_price: float
    session_open: float
    session_high: float
    session_low: float
    session_range: float           # high - low
    price_in_range_pct: float      # 0=at low, 100=at high
    vwap: float
    price_vs_vwap: str             # "above" / "below" / "at"
    opening_gap_pct: float         # vs previous close (if available)
    cumulative_return_pct: float   # open → current
    ad_ratio: float                # accumulation/distribution: +1=full acc, -1=full dist
    momentum_signal: str           # "strong_bull" / "bull" / "neutral" / "bear" / "strong_bear"
    fade_detected: bool            # opening move reversed >50%
    bars_used: int
    verdict: str


def compute(bars: list, *, symbol: str = "", prev_close: float | None = None) -> IntradayMomentum | None:
    """Compute intraday momentum from a list of bars.

    bars: list of bars with .open, .high, .low, .close, .volume
    symbol: symbol name for display
    prev_close: previous session close for gap calculation

    Returns None if fewer than 3 bars or missing OHLCV data.
    """
    if not bars or len(bars) < 3:
        return None

    try:
        opens = [float(b.open) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) if hasattr(b, "volume") else 1.0 for b in bars]
    except Exception:
        return None

    n = len(bars)

    session_open = opens[0]
    session_high = max(highs)
    session_low = min(lows)
    session_range = session_high - session_low
    current_price = closes[-1]

    if session_range <= 0:
        return None

    # Price position in session range
    price_in_range_pct = (current_price - session_low) / session_range * 100

    # VWAP: sum(typical_price × volume) / sum(volume)
    total_vol = sum(volumes)
    if total_vol > 0:
        vwap = sum((highs[i] + lows[i] + closes[i]) / 3 * volumes[i] for i in range(n)) / total_vol
    else:
        vwap = sum(closes) / n

    vwap_tol = vwap * 0.002  # 0.2% tolerance for "at"
    if current_price > vwap + vwap_tol:
        price_vs_vwap = "above"
    elif current_price < vwap - vwap_tol:
        price_vs_vwap = "below"
    else:
        price_vs_vwap = "at"

    # Opening gap
    opening_gap_pct = 0.0
    if prev_close and prev_close > 0:
        opening_gap_pct = (session_open - prev_close) / prev_close * 100

    # Cumulative return
    cumulative_return_pct = (current_price - session_open) / session_open * 100 if session_open > 0 else 0.0

    # Accumulation/Distribution: CLV × volume
    # CLV = ((close - low) - (high - close)) / (high - low)
    ad_total = 0.0
    ad_vol_total = 0.0
    for i in range(n):
        hl = highs[i] - lows[i]
        if hl > 0:
            clv = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        else:
            clv = 0.0
        ad_total += clv * volumes[i]
        ad_vol_total += volumes[i]

    ad_ratio = ad_total / ad_vol_total if ad_vol_total > 0 else 0.0
    ad_ratio = max(-1.0, min(1.0, ad_ratio))

    # Fade detection: did the first-third move reverse?
    fade_detected = False
    if n >= 6:
        first_third = n // 3
        initial_direction = closes[first_third] - session_open
        if abs(initial_direction) > 0:
            full_move = current_price - session_open
            if initial_direction * full_move < 0:
                fade_detected = True
            elif abs(full_move) < abs(initial_direction) * 0.5:
                fade_detected = True

    # Momentum signal
    score = 0
    if price_vs_vwap == "above":
        score += 2
    elif price_vs_vwap == "below":
        score -= 2
    if price_in_range_pct > 70:
        score += 1
    elif price_in_range_pct < 30:
        score -= 1
    if ad_ratio > 0.3:
        score += 1
    elif ad_ratio < -0.3:
        score -= 1
    if cumulative_return_pct > 1.0:
        score += 1
    elif cumulative_return_pct < -1.0:
        score -= 1
    if fade_detected:
        score -= 1

    if score >= 4:
        momentum_signal = "strong_bull"
    elif score >= 2:
        momentum_signal = "bull"
    elif score <= -4:
        momentum_signal = "strong_bear"
    elif score <= -2:
        momentum_signal = "bear"
    else:
        momentum_signal = "neutral"

    # Verdict
    if momentum_signal == "strong_bull":
        verdict = "Strong bullish intraday momentum — price above VWAP, high in range, accumulating"
    elif momentum_signal == "bull":
        verdict = "Bullish intraday bias — generally above VWAP"
    elif momentum_signal == "strong_bear":
        verdict = "Strong bearish intraday momentum — price below VWAP, low in range, distributing"
    elif momentum_signal == "bear":
        verdict = "Bearish intraday bias — generally below VWAP"
    else:
        verdict = "Neutral — balanced intraday price action"

    if fade_detected:
        verdict += " (opening move fading)"

    return IntradayMomentum(
        symbol=symbol,
        current_price=round(current_price, 2),
        session_open=round(session_open, 2),
        session_high=round(session_high, 2),
        session_low=round(session_low, 2),
        session_range=round(session_range, 2),
        price_in_range_pct=round(price_in_range_pct, 1),
        vwap=round(vwap, 2),
        price_vs_vwap=price_vs_vwap,
        opening_gap_pct=round(opening_gap_pct, 2),
        cumulative_return_pct=round(cumulative_return_pct, 2),
        ad_ratio=round(ad_ratio, 3),
        momentum_signal=momentum_signal,
        fade_detected=fade_detected,
        bars_used=n,
        verdict=verdict,
    )
