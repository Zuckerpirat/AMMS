"""Portfolio beta calculation vs a market benchmark.

Computes each position's beta relative to a benchmark (typically SPY)
using OLS regression of daily returns over a lookback window.

Beta interpretation:
  beta > 1.0  — more volatile than market
  beta = 1.0  — moves with market
  beta < 1.0  — defensive, less volatile
  beta < 0.0  — moves opposite to market (rare)

Also computes:
  - alpha: excess return above what beta predicts
  - r_squared: how well the market explains the stock's variance
  - correlation: Pearson correlation with benchmark
  - portfolio_beta: weighted average beta across all positions
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BetaResult:
    symbol: str
    beta: float
    alpha_annualized: float   # annualized alpha (excess return)
    r_squared: float          # 0-1, how much variance explained by market
    correlation: float        # Pearson -1..+1 with benchmark
    weight_pct: float         # portfolio weight %
    beta_contribution: float  # beta × weight (contribution to portfolio beta)
    bars_used: int


@dataclass(frozen=True)
class PortfolioBeta:
    positions: list[BetaResult]
    portfolio_beta: float          # weighted sum of beta × weight
    avg_r_squared: float
    n_positions: int
    benchmark: str
    high_beta_count: int           # count with beta > 1.5
    defensive_count: int           # count with beta < 0.7
    verdict: str


def _ols_beta(stock_returns: list[float], market_returns: list[float]) -> tuple[float, float, float, float]:
    """Return (beta, alpha_daily, r_squared, correlation) from OLS regression."""
    n = len(stock_returns)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0

    # Means
    mx = sum(market_returns) / n
    my = sum(stock_returns) / n

    # Covariance and variance
    cov_xy = sum((market_returns[i] - mx) * (stock_returns[i] - my) for i in range(n))
    var_x = sum((market_returns[i] - mx) ** 2 for i in range(n))
    var_y = sum((stock_returns[i] - my) ** 2 for i in range(n))

    if var_x == 0:
        return 0.0, my, 0.0, 0.0

    beta = cov_xy / var_x
    alpha_daily = my - beta * mx

    # R-squared
    ss_res = sum((stock_returns[i] - (alpha_daily + beta * market_returns[i])) ** 2 for i in range(n))
    ss_tot = var_y
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_sq = max(0.0, min(1.0, r_sq))

    # Correlation
    denom = (var_x * var_y) ** 0.5
    corr = cov_xy / denom if denom > 0 else 0.0
    corr = max(-1.0, min(1.0, corr))

    return beta, alpha_daily, r_sq, corr


def compute(
    positions_bars: dict[str, list],
    benchmark_bars: list,
    *,
    lookback: int = 60,
    benchmark: str = "SPY",
) -> PortfolioBeta | None:
    """Compute beta for each position vs benchmark.

    positions_bars: {symbol: list_of_bars}
    benchmark_bars: list of bars for the benchmark
    lookback: number of bars to use (most recent)
    benchmark: name string for display

    Returns None if benchmark has insufficient data.
    """
    if not benchmark_bars or len(benchmark_bars) < 5:
        return None

    # Build benchmark return series
    bench = benchmark_bars[-lookback:]
    if len(bench) < 5:
        return None

    bench_closes = [float(b.close) for b in bench]
    bench_returns = [
        (bench_closes[i] - bench_closes[i - 1]) / bench_closes[i - 1]
        for i in range(1, len(bench_closes))
    ]

    if not positions_bars:
        return None

    # Compute total portfolio value for weighting
    position_values: dict[str, float] = {}
    for sym, bars in positions_bars.items():
        if bars:
            position_values[sym] = float(bars[-1].close)

    total_value = sum(position_values.values())
    if total_value <= 0:
        return None

    results: list[BetaResult] = []

    for sym, bars in positions_bars.items():
        if not bars or len(bars) < 5:
            continue

        sym_bars = bars[-lookback:]
        closes = [float(b.close) for b in sym_bars]
        stock_returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        # Align lengths
        n = min(len(stock_returns), len(bench_returns))
        if n < 3:
            continue

        s_ret = stock_returns[-n:]
        m_ret = bench_returns[-n:]

        beta, alpha_daily, r_sq, corr = _ols_beta(s_ret, m_ret)

        # Annualize alpha (252 trading days)
        alpha_ann = alpha_daily * 252 * 100  # in percent

        weight = position_values.get(sym, 0) / total_value * 100

        results.append(BetaResult(
            symbol=sym,
            beta=round(beta, 3),
            alpha_annualized=round(alpha_ann, 2),
            r_squared=round(r_sq, 3),
            correlation=round(corr, 3),
            weight_pct=round(weight, 2),
            beta_contribution=round(beta * weight / 100, 4),
            bars_used=n,
        ))

    if not results:
        return None

    # Sort by weight desc
    results.sort(key=lambda r: -r.weight_pct)

    portfolio_beta = sum(r.beta_contribution for r in results)
    avg_r2 = sum(r.r_squared for r in results) / len(results)
    high_beta = sum(1 for r in results if r.beta > 1.5)
    defensive = sum(1 for r in results if r.beta < 0.7)

    # Verdict
    if portfolio_beta > 1.5:
        verdict = "Aggressive — portfolio amplifies market moves significantly"
    elif portfolio_beta > 1.2:
        verdict = "Tilted high-beta — above-market sensitivity"
    elif portfolio_beta > 0.8:
        verdict = "Market-like exposure — close to benchmark"
    elif portfolio_beta > 0.5:
        verdict = "Defensive — lower volatility than market"
    else:
        verdict = "Very defensive or market-neutral"

    return PortfolioBeta(
        positions=results,
        portfolio_beta=round(portfolio_beta, 3),
        avg_r_squared=round(avg_r2, 3),
        n_positions=len(results),
        benchmark=benchmark,
        high_beta_count=high_beta,
        defensive_count=defensive,
        verdict=verdict,
    )
