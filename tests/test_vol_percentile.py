"""Tests for amms.analysis.vol_percentile."""

from __future__ import annotations

import math

import pytest

from amms.data.bars import Bar
from amms.analysis.vol_percentile import VolPercentileResult, compute


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1, close - 1, close, 10_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


def _oscillating(n: int, base: float = 100.0, amp: float = 1.0) -> list[float]:
    """Generate n prices oscillating around base."""
    return [base + amp * math.sin(i * 0.3) for i in range(n)]


class TestCompute:
    def test_returns_none_insufficient_bars(self):
        bars = _bars("AAPL", _oscillating(100))
        assert compute(bars, vol_window=20, history_window=252) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert isinstance(result, VolPercentileResult)

    def test_symbol_preserved(self):
        bars = _bars("TSLA", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_percentile_range(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert 0.0 <= result.percentile <= 100.0

    def test_realized_vol_positive(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.realized_vol > 0

    def test_extreme_regime_high_vol(self):
        """Recent spike in vol → extreme percentile."""
        # Low-vol history then sudden large moves
        prices = _oscillating(280, amp=0.1) + _oscillating(20, amp=5.0)
        bars = _bars("AAPL", prices)
        result = compute(bars, vol_window=10, history_window=100)
        assert result is not None
        assert result.regime in ("elevated", "extreme")
        assert result.percentile > 60

    def test_low_regime_constant_low_vol(self):
        """Current vol lower than most history → low regime."""
        # High-vol history then sudden quiet period
        prices = _oscillating(280, amp=3.0) + _oscillating(20, amp=0.05)
        bars = _bars("AAPL", prices)
        result = compute(bars, vol_window=10, history_window=100)
        assert result is not None
        assert result.regime in ("low", "normal")
        assert result.percentile < 60

    def test_vol_of_vol_positive(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.vol_of_vol >= 0

    def test_mean_vol_positive(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.mean_vol > 0

    def test_median_vol_positive(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.median_vol > 0

    def test_returns_none_zero_prices(self):
        prices = [0.0] * 300
        bars = _bars("AAPL", prices)
        assert compute(bars, vol_window=20, history_window=252) is None

    def test_history_window_in_result(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        assert result.history_window == 252

    def test_vol_window_in_result(self):
        bars = _bars("AAPL", _oscillating(300))
        result = compute(bars, vol_window=15, history_window=200)
        assert result is not None
        assert result.current_vol_window == 15

    def test_smaller_windows_work(self):
        bars = _bars("AAPL", _oscillating(50))
        result = compute(bars, vol_window=5, history_window=20)
        assert result is not None

    def test_regime_normal_for_moderate_vol(self):
        """Uniform GBM vol → percentile not at extreme ends."""
        import random
        random.seed(42)
        # Generate GBM with fixed daily vol: each step is independent
        price = 100.0
        prices = [price]
        for _ in range(299):
            price *= math.exp(random.gauss(0, 0.01))
            prices.append(price)
        bars = _bars("AAPL", prices)
        result = compute(bars, vol_window=20, history_window=252)
        assert result is not None
        # Uniform vol → current percentile should not be at extreme ends
        assert 10 <= result.percentile <= 90
