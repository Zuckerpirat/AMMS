"""Tests for Monte Carlo simulation."""

from __future__ import annotations

from amms.analysis.monte_carlo import MonteCarloResult, simulate


def _returns(win_rate: float = 0.5, avg_win: float = 0.03, avg_loss: float = -0.015, n: int = 50) -> list[float]:
    returns = []
    for i in range(n):
        returns.append(avg_win if i % 2 == 0 else avg_loss)
    return returns


class TestMonteCarlo:
    def test_returns_none_for_too_few_trades(self) -> None:
        assert simulate([0.01, 0.02, -0.01]) is None

    def test_returns_result_for_enough_trades(self) -> None:
        result = simulate(_returns(), n_simulations=100)
        assert result is not None
        assert isinstance(result, MonteCarloResult)

    def test_n_simulations_correct(self) -> None:
        result = simulate(_returns(), n_simulations=200)
        assert result is not None
        assert result.n_simulations == 200

    def test_percentile_order(self) -> None:
        result = simulate(_returns(), n_simulations=200)
        assert result is not None
        assert result.p5_final <= result.p50_final <= result.p95_final

    def test_probabilities_in_range(self) -> None:
        result = simulate(_returns(), n_simulations=200)
        assert result is not None
        assert 0.0 <= result.prob_ruin <= 1.0
        assert 0.0 <= result.prob_positive <= 1.0
        assert 0.0 <= result.prob_20pct_dd <= 1.0

    def test_positive_strategy_mostly_profitable(self) -> None:
        """Strategy with 3:1 R:R and 55% win rate should be mostly profitable."""
        returns = [0.06 if i % 2 == 0 else -0.02 for i in range(50)]
        result = simulate(returns, n_simulations=500)
        assert result is not None
        assert result.prob_positive > 0.5

    def test_losing_strategy_has_ruin(self) -> None:
        """Strategy that always loses."""
        returns = [-0.05] * 50
        result = simulate(returns, n_simulations=100)
        assert result is not None
        assert result.prob_ruin > 0.5

    def test_deterministic_with_seed(self) -> None:
        r1 = simulate(_returns(), n_simulations=100, seed=42)
        r2 = simulate(_returns(), n_simulations=100, seed=42)
        assert r1 is not None and r2 is not None
        assert r1.p50_final == r2.p50_final

    def test_custom_n_trades(self) -> None:
        result = simulate(_returns(), n_simulations=100, n_trades=20)
        assert result is not None
        assert result.n_trades == 20

    def test_drawdown_non_negative(self) -> None:
        result = simulate(_returns(), n_simulations=100)
        assert result is not None
        assert result.median_max_drawdown_pct >= 0.0
        assert result.p95_max_drawdown_pct >= result.median_max_drawdown_pct
