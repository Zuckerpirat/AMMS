"""Tests for amms.analysis.price_forecast."""

from __future__ import annotations

import math

import pytest

from amms.data.bars import Bar
from amms.analysis.price_forecast import PriceForecast, forecast


def _bar(close: float, i: int = 0) -> Bar:
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1, close - 1, close, 10_000)


def _bars(prices: list[float]) -> list[Bar]:
    return [_bar(p, i) for i, p in enumerate(prices)]


class TestForecast:
    def test_returns_none_insufficient(self):
        bars = _bars([100.0] * 20)
        assert forecast(bars, horizon_days=10, n_hist=30) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars([100.0 + i * 0.1 for i in range(35)])
        result = forecast(bars, horizon_days=10, n_hist=30)
        assert result is not None
        assert isinstance(result, PriceForecast)

    def test_symbol_preserved(self):
        bars = _bars([100.0] * 35)
        result = forecast(bars, horizon_days=5, n_hist=30)
        assert result is not None
        assert result.symbol == "SYM"

    def test_current_price_matches_last_close(self):
        prices = [100.0] * 30 + [105.0]
        bars = _bars(prices)
        result = forecast(bars, horizon_days=5, n_hist=30)
        assert result is not None
        assert result.current_price == pytest.approx(105.0, abs=0.01)

    def test_bands_ordered(self):
        """95% CI should be wider than 68% CI."""
        bars = _bars([100.0 + i * 0.3 for i in range(35)])
        result = forecast(bars, horizon_days=10, n_hist=30)
        assert result is not None
        assert result.p95_low <= result.p68_low
        assert result.p68_low <= result.expected
        assert result.expected <= result.p68_high
        assert result.p68_high <= result.p95_high

    def test_wider_ci_for_longer_horizon(self):
        """Longer horizon → wider uncertainty bands."""
        bars = _bars([100.0 + i * 0.3 for i in range(35)])
        r5 = forecast(bars, horizon_days=5, n_hist=30)
        r20 = forecast(bars, horizon_days=20, n_hist=30)
        assert r5 is not None and r20 is not None
        width5 = r5.p95_high - r5.p95_low
        width20 = r20.p95_high - r20.p95_low
        assert width20 > width5

    def test_volatility_positive(self):
        bars = _bars([100.0 + i * 0.3 for i in range(35)])
        result = forecast(bars, horizon_days=10, n_hist=30)
        assert result is not None
        assert result.daily_vol_pct > 0
        assert result.annualized_vol_pct > 0

    def test_annualized_vol_is_daily_times_sqrt252(self):
        bars = _bars([100.0 + i * 0.3 for i in range(35)])
        result = forecast(bars, horizon_days=10, n_hist=30)
        assert result is not None
        # annualized ≈ daily × √252 (rounding may differ by a few basis points)
        expected_ann = result.daily_vol_pct * math.sqrt(252)
        assert result.annualized_vol_pct == pytest.approx(expected_ann, abs=0.1)

    def test_flat_prices_narrow_bands(self):
        """Constant price → zero volatility → very narrow bands."""
        bars = _bars([100.0] * 35)
        result = forecast(bars, horizon_days=10, n_hist=30)
        assert result is not None
        # Bands should be very narrow when std ≈ 0
        assert result.p95_high - result.p95_low < 0.1

    def test_horizon_days_preserved(self):
        bars = _bars([100.0] * 35)
        result = forecast(bars, horizon_days=15, n_hist=30)
        assert result is not None
        assert result.horizon_days == 15

    def test_zero_price_returns_none(self):
        bars = _bars([0.0] * 35)
        assert forecast(bars, horizon_days=10, n_hist=30) is None

    def test_bars_used_correct(self):
        bars = _bars([100.0 + i * 0.1 for i in range(35)])
        result = forecast(bars, horizon_days=5, n_hist=30)
        assert result is not None
        assert result.bars_used == 30
