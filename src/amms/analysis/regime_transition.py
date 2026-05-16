"""Regime Transition Detector.

Identifies when the market regime is transitioning between bull, neutral,
and bear states. Uses a multi-factor signal that combines:

  1. Price vs SMA-50 and SMA-200 (trend filter)
  2. Rate-of-change momentum (20-day)
  3. RSI positioning
  4. Volatility level (ATR% relative to history)

Each factor votes for a regime: +1 = bull, -1 = bear, 0 = neutral.
Total vote determines regime. A "transition" occurs when the current regime
differs from the regime N bars ago.

The transition strength indicates how significant the shift is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSnapshot:
    bar_idx: int
    regime: str     # "bull", "neutral", "bear"
    vote: int       # -4 to +4


@dataclass(frozen=True)
class RegimeTransitionReport:
    symbol: str
    current_regime: str        # "bull", "neutral", "bear"
    previous_regime: str       # 10 bars ago
    transition_detected: bool
    transition_bars_ago: int | None   # when the last transition happened
    transition_direction: str  # "bull_to_bear", "bear_to_bull", "to_neutral", "stable"

    current_vote: int          # -4 to +4
    history: list[RegimeSnapshot]  # last N regime snapshots

    # Current indicator values
    current_price: float
    sma50: float | None
    sma200: float | None
    rsi: float
    roc20: float
    atr_pct: float
    atr_percentile: float  # 0=calmest, 100=most volatile

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
    lc = float(bars[-1].close)
    return atr / lc * 100.0 if lc > 0 else 0.0


def _vote_for_bars(bars: list, idx: int) -> int:
    """Compute regime vote for bar at index idx."""
    sub_bars = bars[:idx + 1]
    if len(sub_bars) < 25:
        return 0
    closes = [float(b.close) for b in sub_bars]
    current = closes[-1]
    if current <= 0:
        return 0

    vote = 0
    # SMA-50
    if len(closes) >= 50:
        sma50 = sum(closes[-50:]) / 50.0
        vote += 1 if current > sma50 else -1
    # SMA-200
    if len(closes) >= 200:
        sma200 = sum(closes[-200:]) / 200.0
        vote += 1 if current > sma200 else -1
    # ROC-20
    if len(closes) >= 21 and closes[-21] > 0:
        roc = (current / closes[-21] - 1.0) * 100.0
        vote += 1 if roc > 2.0 else (-1 if roc < -2.0 else 0)
    # RSI
    rsi_val = _rsi(closes)
    vote += 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)

    return vote


def _regime_from_vote(v: int) -> str:
    if v >= 2:
        return "bull"
    elif v <= -2:
        return "bear"
    return "neutral"


def analyze(bars: list, *, symbol: str = "", history_n: int = 20) -> RegimeTransitionReport | None:
    """Detect regime transitions from bar history.

    bars: bar objects with .close, .high, .low attributes.
    history_n: how many past snapshots to record.
    """
    min_required = 210  # need 200 bars for full SMA-200 computation
    if not bars or len(bars) < 55:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Build recent regime history
    step = max(1, n // history_n)
    history: list[RegimeSnapshot] = []
    for i in range(max(0, n - history_n * step), n, step):
        v = _vote_for_bars(bars, i)
        history.append(RegimeSnapshot(bar_idx=i, regime=_regime_from_vote(v), vote=v))

    # Ensure current bar is last
    current_vote = _vote_for_bars(bars, n - 1)
    current_regime = _regime_from_vote(current_vote)

    # Previous regime: from 10 bars ago
    prev_vote = _vote_for_bars(bars, max(0, n - 11))
    prev_regime = _regime_from_vote(prev_vote)

    transition = current_regime != prev_regime

    # Find when last transition happened
    trans_bars_ago = None
    for snap in reversed(history[:-1]):
        if snap.regime != current_regime:
            trans_bars_ago = n - 1 - snap.bar_idx
            break

    # Direction
    if not transition:
        direction = "stable"
    elif current_regime == "bear":
        direction = "bull_to_bear" if prev_regime == "bull" else "to_bear"
    elif current_regime == "bull":
        direction = "bear_to_bull" if prev_regime == "bear" else "to_bull"
    else:
        direction = "to_neutral"

    # Current indicators
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    rsi_val = _rsi(closes)
    roc20 = (closes[-1] / closes[-21] - 1.0) * 100.0 if len(closes) >= 21 and closes[-21] > 0 else 0.0
    atr = _atr_pct(bars)

    # ATR percentile
    if len(bars) >= 30:
        atr_history = []
        for i in range(14, min(90, len(bars))):
            atr_history.append(_atr_pct(bars[:i + 1]))
        below = sum(1 for a in atr_history if a <= atr)
        atr_pct_rank = below / len(atr_history) * 100.0 if atr_history else 50.0
    else:
        atr_pct_rank = 50.0

    # Verdict
    parts = [f"regime: {current_regime} (vote {current_vote:+d}/4)"]
    if transition:
        parts.append(f"TRANSITION {prev_regime}→{current_regime}")
    else:
        parts.append("no transition")
    if trans_bars_ago is not None:
        parts.append(f"last change {trans_bars_ago} bars ago")

    verdict = "Regime: " + "; ".join(parts) + "."

    return RegimeTransitionReport(
        symbol=symbol,
        current_regime=current_regime,
        previous_regime=prev_regime,
        transition_detected=transition,
        transition_bars_ago=trans_bars_ago,
        transition_direction=direction,
        current_vote=current_vote,
        history=history,
        current_price=round(current, 4),
        sma50=round(sma50, 4) if sma50 else None,
        sma200=round(sma200, 4) if sma200 else None,
        rsi=round(rsi_val, 1),
        roc20=round(roc20, 3),
        atr_pct=round(atr, 3),
        atr_percentile=round(atr_pct_rank, 1),
        bars_used=n,
        verdict=verdict,
    )
