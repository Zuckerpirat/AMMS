"""Tests for amms.analysis.volatility_forecast."""

from __future__ import annotations

import pytest

from amms.analysis.volatility_forecast import VolForecastReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _calm(n: int = 40) -> list[_Bar]:
    """Small random-like oscillations around 100."""
    prices = []
    price = 100.0
    for i in range(n):
        price += 0.05 * (1 if i % 2 == 0 else -1)
        prices.append(_Bar(price))
    return prices


def _volatile(n: int = 40) -> list[_Bar]:
    """Large daily swings."""
    prices = []
    price = 100.0
    for i in range(n):
        price += 3.0 * (1 if i % 2 == 0 else -1)
        prices.append(_Bar(price))
    return prices


def _rising_vol(n: int = 50) -> list[_Bar]:
    """Calm start, then volatile end."""
    bars = []
    price = 100.0
    for i in range(n):
        spread = 0.1 if i < n // 2 else 3.0
        price += spread * (1 if i % 2 == 0 else -1)
        bars.append(_Bar(price))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _calm(8)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _calm(20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, VolForecastReport)


class TestVolatilityValues:
    def test_daily_vol_positive(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.ewma_vol_daily > 0

    def test_annual_vol_positive(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.ewma_vol_annual > 0

    def test_annual_gt_daily(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        # annualised = daily * sqrt(252) ≈ 15.87 × daily
        assert result.ewma_vol_annual > result.ewma_vol_daily

    def test_volatile_higher_than_calm(self):
        calm_result = analyze(_calm(40))
        vol_result = analyze(_volatile(40))
        if calm_result and vol_result:
            assert vol_result.ewma_vol_daily > calm_result.ewma_vol_daily


class TestForecasts:
    def test_vol_1d_positive(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.vol_1d > 0

    def test_vol_5d_positive(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.vol_5d > 0

    def test_vol_10d_positive(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.vol_10d > 0


class TestVaR:
    def test_var_95_negative(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.var_95_1d < 0


class TestPercentile:
    def test_percentile_in_range(self):
        bars = _calm(40)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.vol_percentile <= 100.0

    def test_volatile_higher_percentile(self):
        bars = _rising_vol(50)
        result = analyze(bars)
        assert result is not None
        # After spike, should be in high percentile
        assert result.vol_percentile > 50.0


class TestTrend:
    def test_vol_trend_valid(self):
        bars = _calm(40)
        result = analyze(bars)
        assert result is not None
        assert result.vol_trend in {"rising", "falling", "stable"}

    def test_rising_vol_detected(self):
        bars = _rising_vol(50)
        result = analyze(bars)
        assert result is not None
        # Last bars are volatile — should detect rising
        assert result.vol_trend in {"rising", "stable"}


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_symbol_stored(self):
        bars = _calm(30)
        result = analyze(bars, symbol="SPY")
        assert result is not None
        assert result.symbol == "SPY"

    def test_lambda_stored(self):
        bars = _calm(30)
        result = analyze(bars, lambda_=0.97)
        assert result is not None
        assert result.lambda_ == 0.97


class TestVerdict:
    def test_verdict_present(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_vol(self):
        bars = _calm(30)
        result = analyze(bars)
        assert result is not None
        assert "vol" in result.verdict.lower()
