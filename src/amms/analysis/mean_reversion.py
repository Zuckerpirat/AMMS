"""Mean Reversion Score.

Aggregates multiple indicators to score how "stretched" a price is from
its historical mean. A high score suggests a return to mean is likely.

Inputs:
  - Bollinger %B: 0 = below lower band, 1 = above upper band (stretched when near 0 or 1)
  - Z-score: standard deviations from 20-bar mean
  - RSI deviation from 50: how far RSI has moved from neutral
  - Williams %R: raw overbought/oversold oscillator

Each component contributes 0-25 points to the 0-100 score.

Score interpretation:
  0-20   → price near mean, no mean reversion setup
  20-40  → mild stretch, watch for setup
  40-60  → moderate reversion potential
  60-80  → strong reversion signal
  80-100 → extreme stretch, high-confidence mean reversion expected

Direction: "bullish_reversion" (oversold, buy signal) or "bearish_reversion" (overbought, sell signal)
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class MeanReversionScore:
    symbol: str
    score: float              # 0..100
    direction: str            # "bullish_reversion" | "bearish_reversion" | "neutral"
    verdict: str              # "extreme" | "strong" | "moderate" | "mild" | "none"
    components: dict[str, float]  # individual component scores
    current_price: float
    recommended_action: str


def score(bars: list[Bar], *, n_zscore: int = 20) -> MeanReversionScore | None:
    """Compute mean reversion score for the given bar series.

    Returns None if insufficient data (need at least 20 bars).
    """
    if len(bars) < 20:
        return None

    symbol = bars[0].symbol
    price = bars[-1].close
    components: dict[str, float] = {}
    bullish_points = 0.0
    bearish_points = 0.0

    # Component 1: Bollinger %B (0-25 points)
    try:
        from amms.features.bollinger import bollinger
        bb = bollinger(bars, 20)
        if bb is not None:
            pct_b = bb.pct_b
            if pct_b <= 0.0:
                # Below lower band → oversold → bullish reversion
                stretch = min(abs(pct_b), 1.0)
                pts = 12.5 + stretch * 12.5
                components["bollinger_pct_b"] = round(pts, 1)
                bullish_points += pts
            elif pct_b >= 1.0:
                # Above upper band → overbought → bearish reversion
                stretch = min(pct_b - 1.0, 1.0)
                pts = 12.5 + stretch * 12.5
                components["bollinger_pct_b"] = round(pts, 1)
                bearish_points += pts
            else:
                components["bollinger_pct_b"] = 0.0
    except Exception:
        pass

    # Component 2: Z-score (0-25 points)
    try:
        from amms.features.zscore import zscore as compute_zscore
        z = compute_zscore(bars, n_zscore)
        if z is not None:
            if z < -2.0:
                pts = min((-z - 2.0) * 5.0 + 15.0, 25.0)
                components["zscore"] = round(pts, 1)
                bullish_points += pts
            elif z > 2.0:
                pts = min((z - 2.0) * 5.0 + 15.0, 25.0)
                components["zscore"] = round(pts, 1)
                bearish_points += pts
            elif abs(z) > 1.0:
                pts = (abs(z) - 1.0) * 7.5
                components["zscore"] = round(pts, 1)
                if z < 0:
                    bullish_points += pts
                else:
                    bearish_points += pts
            else:
                components["zscore"] = 0.0
    except Exception:
        pass

    # Component 3: RSI deviation from 50 (0-25 points)
    try:
        from amms.features.momentum import rsi as compute_rsi
        rsi_val = compute_rsi(bars, 14)
        if rsi_val is not None:
            dev = abs(rsi_val - 50.0)
            pts = min(dev / 50.0 * 25.0, 25.0)
            components["rsi_deviation"] = round(pts, 1)
            if rsi_val < 50:
                bullish_points += pts
            else:
                bearish_points += pts
    except Exception:
        pass

    # Component 4: Williams %R (0-25 points)
    try:
        from amms.features.williams_r import williams_r
        wr_result = williams_r(bars, period=14, smooth=1)
        if wr_result is not None:
            wr = wr_result.value
            if wr <= -80.0:
                pts = min((-wr - 80.0) / 20.0 * 12.5 + 12.5, 25.0)
                components["williams_r"] = round(pts, 1)
                bullish_points += pts
            elif wr >= -20.0:
                pts = min((wr + 20.0) / 20.0 * 12.5 + 12.5, 25.0)
                components["williams_r"] = round(pts, 1)
                bearish_points += pts
            else:
                components["williams_r"] = 0.0
    except Exception:
        pass

    total_score = bullish_points + bearish_points

    if bullish_points >= bearish_points:
        direction = "bullish_reversion" if bullish_points > 5 else "neutral"
        dominant = bullish_points
    else:
        direction = "bearish_reversion" if bearish_points > 5 else "neutral"
        dominant = bearish_points

    if dominant >= 80:
        verdict = "extreme"
        action_suffix = "Strong mean-reversion setup."
    elif dominant >= 60:
        verdict = "strong"
        action_suffix = "Likely reversion imminent."
    elif dominant >= 40:
        verdict = "moderate"
        action_suffix = "Monitor for entry."
    elif dominant >= 20:
        verdict = "mild"
        action_suffix = "Early stretch — watch only."
    else:
        verdict = "none"
        action_suffix = "No mean-reversion signal."

    if direction == "bullish_reversion":
        action = f"Oversold stretch detected. {action_suffix} Consider buy."
    elif direction == "bearish_reversion":
        action = f"Overbought stretch detected. {action_suffix} Consider sell."
    else:
        action = "Price near equilibrium. " + action_suffix

    return MeanReversionScore(
        symbol=symbol,
        score=round(total_score, 1),
        direction=direction,
        verdict=verdict,
        components=components,
        current_price=round(price, 4),
        recommended_action=action,
    )
