"""Tests for amms.analysis.volatility_cone."""

from __future__ import annotations

import math

import pytest

from amms.analysis.volatility_cone import VolCone, VolWindow, compute


class _Bar:
    def __init__(self, close):
        self.close = close


def _bars(closes: list[float]) -> list[_Bar]:
    return [_Bar(c) for c in closes]


def _trending(start: float, delta: float, n: int) -> list[_Bar]:
    return _bars([start + i * delta for i in range(n)])


def _volatile(start: float, n: int, amplitude: float = 2.0) -> list[_Bar]:
    """Bars alternating up and down."""
    closes = [start]
    for i in range(n - 1):
        closes.append(closes[-1] + (amplitude if i % 2 == 0 else -amplitude))
    return _bars(closes)


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert compute([]) is None

    def test_returns_none_too_few(self):
        bars = _trending(100.0, 1.0, 20)
        assert compute(bars) is None

    def test_returns_result(self):
        bars = _trending(100.0, 1.0, 120)
        result = compute(bars)
        assert result is not None
        assert isinstance(result, VolCone)


class TestWindows:
    def test_default_windows_present(self):
        bars = _trending(100.0, 1.0, 120)
        result = compute(bars)
        assert result is not None
        window_sizes = [w.window for w in result.windows]
        for expected in [5, 10, 20]:
            assert expected in window_sizes

    def test_custom_windows(self):
        bars = _trending(100.0, 1.0, 150)
        result = compute(bars, windows=[10, 20])
        assert result is not None
        window_sizes = [w.window for w in result.windows]
        assert 10 in window_sizes
        assert 20 in window_sizes

    def test_windows_sorted(self):
        bars = _trending(100.0, 1.0, 120)
        result = compute(bars)
        assert result is not None
        sizes = [w.window for w in result.windows]
        assert sizes == sorted(sizes)

    def test_bars_used(self):
        n = 120
        bars = _trending(100.0, 1.0, n)
        result = compute(bars)
        assert result is not None
        assert result.bars_used == n


class TestHVValues:
    def test_hv_positive(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        for w in result.windows:
            assert w.hv >= 0

    def test_hv_min_leq_median_leq_max(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        for w in result.windows:
            assert w.hv_min <= w.hv_median <= w.hv_max

    def test_percentile_in_range(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        for w in result.windows:
            assert 0.0 <= w.percentile <= 100.0

    def test_25th_leq_75th(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        for w in result.windows:
            assert w.hv_25th <= w.hv_75th

    def test_high_vol_high_percentile(self):
        """Strongly alternating bars → current vol is max → high percentile."""
        # Growing oscillations at end
        closes = [100.0 + i * 0.1 for i in range(100)]
        # Add extreme volatility at end
        for i in range(20):
            closes.append(closes[-1] + (20 if i % 2 == 0 else -20))
        bars = _bars(closes)
        result = compute(bars)
        assert result is not None
        # Short window should have high percentile
        short = result.windows[0]
        assert short.percentile > 50

    def test_low_vol_low_percentile(self):
        """Flat price (tiny moves) at end → low current HV."""
        # High volatility history
        closes = []
        for i in range(100):
            closes.append(100.0 + (5 if i % 2 == 0 else -5))
        # Very flat recent period
        last_price = closes[-1]
        for i in range(20):
            closes.append(last_price + i * 0.001)
        bars = _bars(closes)
        result = compute(bars)
        assert result is not None
        short = result.windows[0]
        assert short.percentile < 50


class TestTermStructure:
    def test_term_structure_valid_value(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        assert result.term_structure in ("normal", "inverted", "flat")

    def test_regime_valid_value(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        assert result.short_term_regime in ("low", "normal", "elevated", "extreme")

    def test_symbol_stored(self):
        bars = _trending(100.0, 1.0, 120)
        result = compute(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_verdict_present(self):
        bars = _volatile(100.0, 120)
        result = compute(bars)
        assert result is not None
        assert len(result.verdict) > 5

    def test_history_window_respected(self):
        """history_window=50 limits how much history is used."""
        bars = _volatile(100.0, 150)
        result = compute(bars, history_window=50)
        assert result is not None
        # Should still produce results
        assert len(result.windows) > 0
