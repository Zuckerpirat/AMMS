"""Portfolio weight optimizer.

Computes target allocation weights for a set of symbols using one of:
  - equal_weight: 1/N per symbol
  - momentum: weights proportional to composite momentum score
  - inverse_vol: weights inversely proportional to realized volatility
                 (risk parity approximation)

Lives in the analysis layer — pure calculation, no trade execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WeightMode = str  # "equal_weight" | "momentum" | "inverse_vol"


@dataclass(frozen=True)
class AllocationResult:
    symbol: str
    weight: float       # 0..1 (fraction of portfolio)
    weight_pct: float   # weight * 100
    mode: str
    score: float | None = None  # underlying score used for weighting


def optimize(
    symbols: list[str],
    data,
    *,
    mode: WeightMode = "equal_weight",
) -> list[AllocationResult]:
    """Compute target weights for the given symbols.

    Returns a list sorted by weight descending. All weights sum to 1.0.
    Returns empty list if no symbols or all data is missing.
    """
    if not symbols:
        return []

    if mode == "equal_weight":
        w = 1.0 / len(symbols)
        return [
            AllocationResult(sym, w, w * 100, mode)
            for sym in symbols
        ]

    if mode == "momentum":
        return _momentum_weights(symbols, data)

    if mode == "inverse_vol":
        return _inverse_vol_weights(symbols, data)

    raise ValueError(f"Unknown mode: {mode!r}. Use equal_weight, momentum, or inverse_vol")


def _momentum_weights(symbols: list[str], data) -> list[AllocationResult]:
    from amms.analysis.momentum_scan import scan
    results = scan(symbols, data, top_n=len(symbols))
    if not results:
        # Fallback to equal weight
        w = 1.0 / len(symbols)
        return [AllocationResult(sym, w, w * 100, "momentum") for sym in symbols]

    total_score = sum(r.score for r in results) or 1.0
    seen = {r.symbol for r in results}
    out = []
    for r in results:
        w = r.score / total_score
        out.append(AllocationResult(r.symbol, w, w * 100, "momentum", score=r.score))
    # symbols with no data get zero weight
    for sym in symbols:
        if sym not in seen:
            out.append(AllocationResult(sym, 0.0, 0.0, "momentum", score=None))
    out.sort(key=lambda x: x.weight, reverse=True)
    return out


def _inverse_vol_weights(symbols: list[str], data) -> list[AllocationResult]:
    from amms.features.volatility import realized_vol

    inv_vols: dict[str, float] = {}
    for sym in symbols:
        try:
            bars = data.get_bars(sym, limit=25)
            rv = realized_vol(bars, 20)
        except Exception:
            rv = None
        if rv and rv > 0:
            inv_vols[sym] = 1.0 / rv
        else:
            inv_vols[sym] = 1.0  # neutral if no data

    total = sum(inv_vols.values()) or 1.0
    out = []
    for sym in symbols:
        w = inv_vols[sym] / total
        out.append(AllocationResult(sym, w, w * 100, "inverse_vol"))
    out.sort(key=lambda x: x.weight, reverse=True)
    return out


def format_allocation(results: list[AllocationResult], equity: float = 0.0) -> str:
    """Format allocation results as a Telegram-friendly string."""
    if not results:
        return "No allocation computed."
    mode = results[0].mode if results else "?"
    lines = [f"Target allocation ({mode.replace('_', '-')}):"]
    for r in results:
        bar = "█" * int(r.weight_pct / 2)
        dollar = f"  ${equity * r.weight:,.0f}" if equity > 0 else ""
        score_str = f"  score {r.score:.0f}" if r.score is not None else ""
        lines.append(
            f"  {r.symbol:<6}  {r.weight_pct:5.1f}%  [{bar:<10}]{score_str}{dollar}"
        )
    return "\n".join(lines)
