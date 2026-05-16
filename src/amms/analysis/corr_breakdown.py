"""Correlation Breakdown Monitor.

Detects when the correlation structure between portfolio positions has
changed significantly vs. a baseline period — a common sign that:
  - Tail risk / panic mode is beginning (all assets start moving together)
  - Market regime has shifted (previously correlated assets decorrelate)
  - A position has changed behaviour (earnings, sector rotation, etc.)

Algorithm:
  1. Compute pairwise Pearson correlation in baseline window (older half)
  2. Compute pairwise Pearson correlation in recent window (newer half)
  3. For each pair: measure |Δcorr| = |recent - baseline|
  4. Flag pairs with |Δcorr| > threshold as "broken" correlations
  5. Compute average correlation in each window (portfolio-level measure)
  6. If avg recent correlation >> avg baseline: correlation surge (tail risk)

A "correlation surge" is the most dangerous scenario: previously
diversified positions all move together, eliminating diversification
exactly when you need it most (crisis correlation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairChange:
    sym1: str
    sym2: str
    baseline_corr: float
    recent_corr: float
    delta: float          # recent - baseline
    kind: str             # "convergence" / "divergence" / "stable"


@dataclass(frozen=True)
class CorrBreakdownReport:
    symbols: list[str]
    pairs: list[PairChange]           # all pairs, sorted by |delta| desc
    broken_pairs: list[PairChange]    # pairs with |delta| > threshold
    avg_baseline_corr: float          # portfolio avg correlation (baseline)
    avg_recent_corr: float            # portfolio avg correlation (recent)
    corr_surge: bool                  # recent corr significantly higher
    corr_collapse: bool               # recent corr significantly lower
    n_bars_baseline: int
    n_bars_recent: int
    surge_threshold: float
    verdict: str


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs))
    sy = math.sqrt(sum((v - my) ** 2 for v in ys))
    if sx <= 0 or sy <= 0:
        return None
    return cov / (sx * sy)


def _returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]


def analyze(
    bars_by_symbol: dict[str, list],
    *,
    breakdown_threshold: float = 0.3,
    surge_threshold: float = 0.2,
) -> CorrBreakdownReport | None:
    """Detect correlation changes across portfolio positions.

    bars_by_symbol: {symbol: list[Bar]} — each Bar needs .close.
    breakdown_threshold: |Δcorr| above which a pair is flagged (default 0.3).
    surge_threshold: Δavg_corr above which a surge is declared (default 0.2).
    """
    if len(bars_by_symbol) < 2:
        return None

    # Compute return series for each symbol; only keep symbols with enough data
    rets: dict[str, list[float]] = {}
    for sym, bars in bars_by_symbol.items():
        if len(bars) < 10:
            continue
        try:
            closes = [float(b.close) for b in bars]
        except Exception:
            continue
        r = _returns(closes)
        if len(r) >= 10:
            rets[sym] = r

    if len(rets) < 2:
        return None

    # Align return series to same length
    min_len = min(len(r) for r in rets.values())
    aligned = {sym: r[-min_len:] for sym, r in rets.items()}
    symbols = sorted(aligned.keys())

    split = min_len // 2
    if split < 3:
        return None

    n_base = split
    n_recent = min_len - split

    base_slices = {s: aligned[s][:split] for s in symbols}
    recent_slices = {s: aligned[s][split:] for s in symbols}

    pairs: list[PairChange] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            bc = _pearson(base_slices[s1], base_slices[s2])
            rc = _pearson(recent_slices[s1], recent_slices[s2])
            if bc is None or rc is None:
                continue
            delta = rc - bc
            if abs(delta) > 0.5:
                kind = "convergence" if delta > 0 else "divergence"
            else:
                kind = "stable"
            pairs.append(PairChange(
                sym1=s1,
                sym2=s2,
                baseline_corr=round(bc, 3),
                recent_corr=round(rc, 3),
                delta=round(delta, 3),
                kind=kind,
            ))

    if not pairs:
        return None

    pairs_sorted = sorted(pairs, key=lambda p: -abs(p.delta))
    broken = [p for p in pairs_sorted if abs(p.delta) > breakdown_threshold]

    # Portfolio-level average correlation
    avg_base = sum(abs(p.baseline_corr) for p in pairs) / len(pairs)
    avg_recent = sum(abs(p.recent_corr) for p in pairs) / len(pairs)
    avg_delta = avg_recent - avg_base

    surge = avg_delta > surge_threshold
    collapse = avg_delta < -surge_threshold

    # Verdict
    if surge:
        cond = f"CORRELATION SURGE detected — avg corr +{avg_delta:.2f} (diversification breaking down)"
    elif collapse:
        cond = f"Correlation collapse detected — avg corr {avg_delta:.2f} (assets decorrelating)"
    else:
        cond = f"Correlation structure stable (avg Δ {avg_delta:+.2f})"

    broken_note = (
        f"  {len(broken)} pair(s) show breakdown (|Δ|>{breakdown_threshold}): "
        + ", ".join(f"{p.sym1}/{p.sym2}({p.delta:+.2f})" for p in broken[:3])
        if broken else ""
    )
    verdict = f"{cond}. Baseline avg: {avg_base:.2f}, Recent avg: {avg_recent:.2f}.{broken_note}"

    return CorrBreakdownReport(
        symbols=symbols,
        pairs=pairs_sorted,
        broken_pairs=broken,
        avg_baseline_corr=round(avg_base, 3),
        avg_recent_corr=round(avg_recent, 3),
        corr_surge=surge,
        corr_collapse=collapse,
        n_bars_baseline=n_base,
        n_bars_recent=n_recent,
        surge_threshold=surge_threshold,
        verdict=verdict,
    )
