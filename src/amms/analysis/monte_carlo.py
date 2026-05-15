"""Monte Carlo simulation for equity curves.

Simulates N random equity curve paths based on historical trade statistics.
Used to estimate:
  - Probability of reaching a drawdown threshold
  - Expected range of outcomes (5th/50th/95th percentile)
  - Risk of ruin (equity < 50% of start)

Input: list of historical trade returns (as fractions: +0.03 = +3%)
Output: simulation statistics across N paths of length M trades.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloResult:
    n_simulations: int
    n_trades: int
    initial_equity: float
    # Equity percentiles at end of simulation
    p5_final: float     # 5th percentile final equity
    p50_final: float    # median final equity
    p95_final: float    # 95th percentile final equity
    # Drawdown stats
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float   # 95th percentile worst drawdown
    prob_ruin: float              # P(equity < 50% of start)
    prob_20pct_dd: float          # P(drawdown exceeds 20%)
    prob_positive: float          # P(final equity > initial)


def simulate(
    trade_returns: list[float],
    *,
    n_simulations: int = 1000,
    n_trades: int | None = None,
    initial_equity: float = 100_000.0,
    seed: int | None = 42,
) -> MonteCarloResult | None:
    """Run Monte Carlo simulation by randomly sampling from trade_returns.

    trade_returns: list of historical per-trade returns as fractions
    n_trades: number of trades per simulation (default: len(trade_returns))
    n_simulations: number of paths to simulate
    seed: random seed for reproducibility
    """
    if len(trade_returns) < 5:
        return None

    if n_trades is None:
        n_trades = len(trade_returns)

    rng = random.Random(seed)

    final_equities: list[float] = []
    max_drawdowns: list[float] = []
    ruin_count = 0
    dd_20_count = 0
    positive_count = 0

    for _ in range(n_simulations):
        equity = initial_equity
        peak = equity
        max_dd = 0.0

        for _ in range(n_trades):
            r = rng.choice(trade_returns)
            equity *= (1 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        final_equities.append(equity)
        max_drawdowns.append(max_dd * 100)

        if equity < initial_equity * 0.5:
            ruin_count += 1
        if max_dd * 100 > 20.0:
            dd_20_count += 1
        if equity > initial_equity:
            positive_count += 1

    final_equities.sort()
    max_drawdowns.sort()
    n = n_simulations

    def _percentile(lst: list[float], pct: float) -> float:
        idx = int(len(lst) * pct / 100)
        return lst[min(idx, len(lst) - 1)]

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_equity=initial_equity,
        p5_final=round(_percentile(final_equities, 5), 2),
        p50_final=round(_percentile(final_equities, 50), 2),
        p95_final=round(_percentile(final_equities, 95), 2),
        median_max_drawdown_pct=round(_percentile(max_drawdowns, 50), 2),
        p95_max_drawdown_pct=round(_percentile(max_drawdowns, 95), 2),
        prob_ruin=round(ruin_count / n, 3),
        prob_20pct_dd=round(dd_20_count / n, 3),
        prob_positive=round(positive_count / n, 3),
    )
