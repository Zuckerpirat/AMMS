"""Momentum scanner.

Ranks a list of symbols by a composite momentum score computed from
multiple technical signals. Lives in the analysis layer — pure scoring,
no trade decisions.

Score components (each contributes 0..25 points):
  1. RSI-14 reversal: oversold (RSI<40) scores highest; overbought (RSI>70) = 0
  2. EMA trend: price above both EMA-20 and EMA-50 = 25; above one = 12; neither = 0
  3. 20-day momentum: top quartile = 25; bottom quartile = 0; linear in between
  4. ATR-relative volume: low-volatility strong moves score higher

Total range: 0..100 (higher = stronger bullish momentum signal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanResult:
    symbol: str
    score: float        # 0..100
    rsi: float | None
    ema_trend: str      # "strong_bull" | "weak_bull" | "bear" | "unknown"
    momentum_20d: float | None  # percent
    reason: str


def scan(
    symbols: list[str],
    data,
    *,
    top_n: int = 10,
) -> list[ScanResult]:
    """Score and rank symbols by momentum. Returns top_n results sorted by score."""
    from amms.features.momentum import ema as compute_ema, n_day_return, rsi as compute_rsi
    from amms.features.volatility import realized_vol

    results: list[ScanResult] = []
    momentums: list[float] = []

    raw: list[tuple[str, float | None, str, float | None, list[float]]] = []

    for sym in symbols:
        try:
            bars = data.get_bars(sym, limit=60)
        except Exception:
            bars = []
        if not bars:
            continue

        r = compute_rsi(bars, 14)
        e20 = compute_ema(bars, 20)
        e50 = compute_ema(bars, 50)
        price = bars[-1].close
        mom = n_day_return(bars, 20)

        if e20 and e50 and price:
            if price > e20 > e50:
                ema_trend = "strong_bull"
            elif price > e20 or e20 > e50:
                ema_trend = "weak_bull"
            else:
                ema_trend = "bear"
        else:
            ema_trend = "unknown"

        raw.append((sym, r, ema_trend, mom, bars))
        if mom is not None:
            momentums.append(mom)

    if not raw:
        return []

    # Normalise momentum into quartile score
    sorted_moms = sorted(momentums)
    n_mom = len(sorted_moms)

    def _mom_score(mom: float | None) -> float:
        if mom is None or not sorted_moms:
            return 12.5  # neutral
        rank = sum(1 for m in sorted_moms if m <= mom) / n_mom
        return rank * 25.0

    for sym, r, ema_trend, mom, bars in raw:
        score = 0.0
        parts: list[str] = []

        # RSI component
        if r is not None:
            if r < 30:
                rsi_sc = 25.0
            elif r < 40:
                rsi_sc = 20.0
            elif r < 50:
                rsi_sc = 15.0
            elif r < 60:
                rsi_sc = 10.0
            elif r < 70:
                rsi_sc = 5.0
            else:
                rsi_sc = 0.0
            score += rsi_sc
            parts.append(f"RSI={r:.0f}")

        # EMA component
        ema_sc = {"strong_bull": 25.0, "weak_bull": 12.0, "bear": 0.0, "unknown": 5.0}
        score += ema_sc.get(ema_trend, 5.0)
        parts.append(ema_trend)

        # Momentum component
        ms = _mom_score(mom)
        score += ms
        if mom is not None:
            parts.append(f"mom={mom*100:.1f}%")

        # Volatility penalty: high vol reduces score
        try:
            rv = realized_vol(bars, 20)
            if rv and rv > 0.5:
                score -= 5.0
                parts.append("hiVol")
        except Exception:
            pass

        score = max(0.0, min(100.0, score))
        results.append(ScanResult(
            symbol=sym,
            score=round(score, 1),
            rsi=r,
            ema_trend=ema_trend,
            momentum_20d=mom,
            reason=", ".join(parts),
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]
