"""Tests for amms.analysis.beta_calculator."""

from __future__ import annotations

import math

import pytest

from amms.analysis.beta_calculator import BetaResult, PortfolioBeta, compute


class _Bar:
    def __init__(self, close):
        self.close = close


def _bars(closes: list[float]) -> list[_Bar]:
    return [_Bar(c) for c in closes]


def _trending(start: float, delta: float, n: int) -> list[_Bar]:
    return _bars([start + i * delta for i in range(n)])


def _market_bars(n: int = 65) -> list[_Bar]:
    """Synthetic market: 1% daily growth."""
    return _trending(100.0, 1.0, n)


class TestEdgeCases:
    def test_none_empty_benchmark(self):
        pos = {"AAPL": _trending(150.0, 1.5, 65)}
        assert compute(pos, []) is None

    def test_none_short_benchmark(self):
        pos = {"AAPL": _trending(150.0, 1.5, 65)}
        assert compute(pos, _market_bars(3)) is None

    def test_none_empty_positions(self):
        bench = _market_bars(65)
        assert compute({}, bench) is None

    def test_none_short_position(self):
        bench = _market_bars(65)
        pos = {"AAPL": _bars([100.0, 101.0])}
        assert compute(pos, bench) is None

    def test_returns_portfolio_beta(self):
        bench = _market_bars(65)
        pos = {"AAPL": _trending(150.0, 1.5, 65)}
        result = compute(pos, bench)
        assert result is not None
        assert isinstance(result, PortfolioBeta)


class TestBetaValues:
    def test_high_beta_stock(self):
        """Stock moving 2× market returns → beta ≈ 2."""
        n = 65
        market = [100.0 + i for i in range(n)]
        # Stock returns = 2× market returns → beta=2
        stock = [100.0]
        for i in range(1, n):
            mret = (market[i] - market[i - 1]) / market[i - 1]
            stock.append(stock[-1] * (1 + 2 * mret))

        bench = _bars(market)
        pos = {"HB": _bars(stock)}
        result = compute(pos, bench)
        assert result is not None
        assert result.positions[0].beta == pytest.approx(2.0, abs=0.15)

    def test_low_beta_stock(self):
        """Stock moving 0.5× market returns → beta ≈ 0.5."""
        n = 65
        market = [100.0 + i for i in range(n)]
        stock = [100.0]
        for i in range(1, n):
            mret = (market[i] - market[i - 1]) / market[i - 1]
            stock.append(stock[-1] * (1 + 0.5 * mret))

        bench = _bars(market)
        pos = {"LB": _bars(stock)}
        result = compute(pos, bench)
        assert result is not None
        assert result.positions[0].beta == pytest.approx(0.5, abs=0.1)

    def test_beta_one_for_identical_stock(self):
        """Stock identical to market → beta = 1, r_squared = 1."""
        market = [100.0 + i for i in range(65)]
        bench = _bars(market)
        pos = {"MKT": _bars(market[:])}
        result = compute(pos, bench)
        assert result is not None
        pos_result = result.positions[0]
        assert pos_result.beta == pytest.approx(1.0, abs=0.05)
        assert pos_result.r_squared == pytest.approx(1.0, abs=0.05)

    def test_portfolio_beta_single_position(self):
        """Single position at 100% weight → portfolio beta = position beta."""
        n = 65
        market = [100.0 + i for i in range(n)]
        stock = [100.0]
        for i in range(1, n):
            mret = (market[i] - market[i - 1]) / market[i - 1]
            stock.append(stock[-1] * (1 + 1.5 * mret))

        bench = _bars(market)
        pos = {"X": _bars(stock)}
        result = compute(pos, bench)
        assert result is not None
        assert result.portfolio_beta == pytest.approx(result.positions[0].beta, abs=0.01)

    def test_r_squared_near_one_for_correlated(self):
        """Stock linearly correlated → R² should be high."""
        n = 65
        market = [100.0 + i for i in range(n)]
        stock = [200.0 + i * 1.5 for i in range(n)]
        bench = _bars(market)
        pos = {"X": _bars(stock)}
        result = compute(pos, bench)
        assert result is not None
        assert result.positions[0].r_squared > 0.9

    def test_correlation_one_for_identical(self):
        """Identical price series → correlation = 1."""
        market = [100.0 + i for i in range(65)]
        bench = _bars(market)
        pos = {"X": _bars(market[:])}
        result = compute(pos, bench)
        assert result is not None
        assert result.positions[0].correlation == pytest.approx(1.0, abs=0.05)


class TestPortfolioStats:
    def _two_position_result(self):
        n = 65
        market = [100.0 + i for i in range(n)]
        bench = _bars(market)
        stock_a = [100.0]
        stock_b = [200.0]
        for i in range(1, n):
            mret = (market[i] - market[i - 1]) / market[i - 1]
            stock_a.append(stock_a[-1] * (1 + 2.0 * mret))
            stock_b.append(stock_b[-1] * (1 + 0.5 * mret))
        pos = {"A": _bars(stock_a), "B": _bars(stock_b)}
        return compute(pos, bench)

    def test_n_positions(self):
        result = self._two_position_result()
        assert result is not None
        assert result.n_positions == 2

    def test_high_beta_count(self):
        result = self._two_position_result()
        assert result is not None
        assert result.high_beta_count >= 1

    def test_defensive_count(self):
        result = self._two_position_result()
        assert result is not None
        assert result.defensive_count >= 1

    def test_verdict_present(self):
        result = self._two_position_result()
        assert result is not None
        assert len(result.verdict) > 5

    def test_benchmark_label(self):
        bench = _market_bars(65)
        pos = {"X": _trending(100.0, 1.0, 65)}
        result = compute(pos, bench, benchmark="SPY")
        assert result is not None
        assert result.benchmark == "SPY"

    def test_beta_contribution_sums_to_portfolio_beta(self):
        n = 65
        market = [100.0 + i for i in range(n)]
        bench = _bars(market)
        stock_a = [100.0 + i * 2 for i in range(n)]
        stock_b = [100.0 + i * 0.5 for i in range(n)]
        pos = {"A": _bars(stock_a), "B": _bars(stock_b)}
        result = compute(pos, bench)
        assert result is not None
        total_contrib = sum(r.beta_contribution for r in result.positions)
        assert total_contrib == pytest.approx(result.portfolio_beta, abs=0.01)

    def test_lookback_respected(self):
        """Using lookback=20 uses fewer bars."""
        bench = _market_bars(80)
        pos = {"X": _trending(100.0, 1.0, 80)}
        result = compute(pos, bench, lookback=20)
        assert result is not None
        # bars_used should be <= lookback-1 (returns from n prices)
        assert result.positions[0].bars_used <= 20

    def test_avg_r_squared_in_range(self):
        bench = _market_bars(65)
        pos = {"X": _trending(100.0, 1.0, 65)}
        result = compute(pos, bench)
        assert result is not None
        assert 0.0 <= result.avg_r_squared <= 1.0

    def test_positions_sorted_by_weight_desc(self):
        n = 65
        market = [100.0 + i for i in range(n)]
        bench = _bars(market)
        # A is cheaper, B is more expensive → B has higher weight
        pos = {
            "A": _bars([10.0 + i * 0.1 for i in range(n)]),
            "B": _bars([200.0 + i * 2.0 for i in range(n)]),
        }
        result = compute(pos, bench)
        assert result is not None
        weights = [r.weight_pct for r in result.positions]
        assert weights == sorted(weights, reverse=True)
