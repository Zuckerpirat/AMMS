"""Heikin-Ashi trend analysis.

Converts standard OHLC candles to Heikin-Ashi (HA) candles, which
smooth noise and make trends easier to see.

HA formulas:
  HA_Close = (O + H + L + C) / 4
  HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
  HA_High  = max(H, HA_Open, HA_Close)
  HA_Low   = min(L, HA_Open, HA_Close)

Bullish HA candle: HA_Close > HA_Open (no lower wick = strong trend)
Bearish HA candle: HA_Close < HA_Open (no upper wick = strong trend)

Interprets:
  - Consecutive bullish HA candles = uptrend strength
  - Consecutive bearish HA candles = downtrend strength
  - Mixed = consolidation/reversal area
  - Doji (open ≈ close) = indecision/potential reversal
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HACandle:
    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float
    bullish: bool      # ha_close > ha_open
    no_lower_wick: bool  # wick < 5% of body (strong bull signal)
    no_upper_wick: bool  # wick < 5% of body (strong bear signal)


@dataclass(frozen=True)
class HeikinAshiReport:
    symbol: str
    ha_candles: list[HACandle]    # last N HA candles
    consecutive_bull: int         # current run of bullish HA candles
    consecutive_bear: int         # current run of bearish HA candles
    trend: str                    # "up" / "down" / "reversal" / "consolidating"
    trend_strength: float         # 0-100 based on consecutive count + wick analysis
    strong_candles: int           # no-wick candles in the current run
    current_ha_open: float
    current_ha_close: float
    bars_used: int
    verdict: str


def _convert(bars: list) -> list[HACandle]:
    """Convert standard bars to Heikin-Ashi candles."""
    if not bars:
        return []
    ha = []
    prev_ha_open = (float(bars[0].open if hasattr(bars[0], 'open') else bars[0].close) +
                    float(bars[0].close)) / 2
    prev_ha_close = (float(bars[0].high if hasattr(bars[0], 'high') else bars[0].close) +
                     float(bars[0].low if hasattr(bars[0], 'low') else bars[0].close) +
                     float(bars[0].close) + prev_ha_open) / 4

    for b in bars:
        try:
            o = float(b.open) if hasattr(b, 'open') else float(b.close)
            h = float(b.high)
            l = float(b.low)
            c = float(b.close)
        except Exception:
            continue

        ha_close = (o + h + l + c) / 4
        ha_open = (prev_ha_open + prev_ha_close) / 2
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)

        body = abs(ha_close - ha_open)
        total_range = ha_high - ha_low
        lower_wick = min(ha_open, ha_close) - ha_low
        upper_wick = ha_high - max(ha_open, ha_close)
        wick_threshold = body * 0.05 if body > 0 else 0.001

        ha.append(HACandle(
            ha_open=round(ha_open, 4),
            ha_high=round(ha_high, 4),
            ha_low=round(ha_low, 4),
            ha_close=round(ha_close, 4),
            bullish=ha_close > ha_open,
            no_lower_wick=lower_wick <= wick_threshold,
            no_upper_wick=upper_wick <= wick_threshold,
        ))
        prev_ha_open = ha_open
        prev_ha_close = ha_close

    return ha


def analyze(bars: list, *, symbol: str = "", lookback: int = 20) -> HeikinAshiReport | None:
    """Analyze Heikin-Ashi trend from bars.

    bars: list[Bar] with .high .low .close — at least 5 bars.
    symbol: ticker for display.
    lookback: number of recent HA candles to analyze.
    """
    if not bars or len(bars) < 5:
        return None

    try:
        ha_all = _convert(bars)
    except Exception:
        return None

    if not ha_all:
        return None

    recent = ha_all[-lookback:]
    current = recent[-1]

    # Count consecutive runs
    bull_run = 0
    bear_run = 0
    for c in reversed(recent):
        if c.bullish:
            if bear_run == 0:
                bull_run += 1
            else:
                break
        else:
            if bull_run == 0:
                bear_run += 1
            else:
                break

    # Strong candles (no opposite wick) in current run
    if bull_run > 0:
        strong = sum(1 for c in reversed(recent[:bull_run]) if c.no_lower_wick)
    else:
        strong = sum(1 for c in reversed(recent[:bear_run]) if c.no_upper_wick)

    # Classify trend
    if bull_run >= 3:
        trend = "up"
    elif bear_run >= 3:
        trend = "down"
    elif bull_run == 1 and bear_run == 0 and current.bullish:
        trend = "reversal"  # single bullish after bearish run
    elif bear_run == 1 and bull_run == 0 and not current.bullish:
        trend = "reversal"
    else:
        trend = "consolidating"

    # Trend strength based on run length and wick quality
    run_length = max(bull_run, bear_run)
    strength = min(100.0, run_length / lookback * 100 + strong * 10)

    trend_desc = {
        "up": f"Uptrend — {bull_run} consecutive bullish HA candles",
        "down": f"Downtrend — {bear_run} consecutive bearish HA candles",
        "reversal": "Potential reversal signal detected",
        "consolidating": "Consolidation / mixed signals",
    }.get(trend, trend)

    strong_note = f" ({strong} no-wick candles = high conviction)" if strong > 0 else ""
    verdict = f"{trend_desc}{strong_note}. Trend strength: {strength:.0f}/100."

    return HeikinAshiReport(
        symbol=symbol,
        ha_candles=recent[-10:],  # last 10 HA candles
        consecutive_bull=bull_run,
        consecutive_bear=bear_run,
        trend=trend,
        trend_strength=round(strength, 1),
        strong_candles=strong,
        current_ha_open=current.ha_open,
        current_ha_close=current.ha_close,
        bars_used=len(bars),
        verdict=verdict,
    )
