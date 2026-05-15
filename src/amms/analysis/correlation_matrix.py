"""Portfolio correlation matrix.

Computes the pairwise Pearson correlation of daily returns between all held
positions. Identifies highly correlated clusters (concentration risk) and
uncorrelated positions (diversification benefit).

Correlation ranges:
  |r| >= 0.8 → very high correlation (nearly same movement)
  |r| >= 0.6 → high correlation (significant overlap)
  |r| >= 0.4 → moderate correlation (some overlap)
  |r| <  0.4 → low correlation (diversified)

Negative correlation is desirable for hedging.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrPair:
    sym1: str
    sym2: str
    correlation: float
    level: str   # "very_high" | "high" | "moderate" | "low" | "negative"


@dataclass(frozen=True)
class CorrelationMatrix:
    symbols: list[str]
    pairs: list[CorrPair]          # all pairwise correlations
    high_corr_pairs: list[CorrPair]  # |r| >= 0.6 (concentration risk)
    avg_correlation: float         # mean |r| across all pairs
    diversification_score: float   # 0..100 (100 = perfectly diversified)
    n_bars: int                    # bars used per symbol


def compute(broker, data, *, n: int = 30) -> CorrelationMatrix | None:
    """Compute the correlation matrix for all open positions.

    n: lookback period for return computation
    Returns None if fewer than 2 positions or insufficient data.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if len(positions) < 2:
        return None

    symbols = [p.symbol for p in positions]
    returns: dict[str, list[float]] = {}

    for sym in symbols:
        try:
            bars = data.get_bars(sym, limit=n + 5)
        except Exception:
            continue
        if len(bars) < 3:
            continue
        closes = [b.close for b in bars[-(n + 1):]]
        if len(closes) < 3:
            continue
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(rets) >= 2:
            returns[sym] = rets

    present = [s for s in symbols if s in returns]
    if len(present) < 2:
        return None

    n_bars_used = min(len(returns[s]) for s in present)

    pairs: list[CorrPair] = []
    for i, s1 in enumerate(present):
        for s2 in present[i + 1:]:
            r = _pearson(returns[s1], returns[s2])
            if r is None:
                continue
            abs_r = abs(r)
            if abs_r >= 0.8:
                level = "very_high"
            elif abs_r >= 0.6:
                level = "high"
            elif abs_r >= 0.4:
                level = "moderate"
            elif r < -0.2:
                level = "negative"
            else:
                level = "low"
            pairs.append(CorrPair(sym1=s1, sym2=s2, correlation=round(r, 3), level=level))

    high_corr = [p for p in pairs if p.level in {"very_high", "high"}]
    # Use only positive correlations for avg — negative correlation benefits diversification
    avg_corr = sum(abs(p.correlation) for p in pairs) / len(pairs) if pairs else 0.0
    # Diversification: consider negative correlation as beneficial (caps at 0 penalty)
    positive_avg = sum(max(p.correlation, 0.0) for p in pairs) / len(pairs) if pairs else 0.0
    diversity_score = max(0.0, (1.0 - positive_avg) * 100)

    return CorrelationMatrix(
        symbols=present,
        pairs=sorted(pairs, key=lambda p: abs(p.correlation), reverse=True),
        high_corr_pairs=high_corr,
        avg_correlation=round(avg_corr, 3),
        diversification_score=round(diversity_score, 1),
        n_bars=n_bars_used,
    )


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da * db > 0 else None
