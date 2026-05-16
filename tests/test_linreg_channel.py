"""Tests for amms.analysis.linreg_channel."""

from __future__ import annotations

import pytest

from amms.analysis.linreg_channel import LRCBand, LRCReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _flat(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _perfect_uptrend(n: int = 60, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    """Perfectly linear uptrend — R² should be ~1.0."""
    return [_Bar(start + i * step) for i in range(n)]


def _noisy_uptrend(n: int = 60, start: float = 50.0, step: float = 0.5) -> list[_Bar]:
    """Uptrend with noise."""
    bars = []
    price = start
    for i in range(n):
        noise = 2.0 * (1 if i % 3 == 0 else -0.5)
        bars.append(_Bar(price + noise))
        price += step
    return bars


def _downtrend(n: int = 60, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(5)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(55))
        assert result is not None
        assert isinstance(result, LRCReport)


class TestSlope:
    def test_uptrend_positive_slope(self):
        result = analyze(_perfect_uptrend(60))
        assert result is not None
        assert result.slope > 0

    def test_downtrend_negative_slope(self):
        result = analyze(_downtrend(60))
        assert result is not None
        assert result.slope < 0

    def test_flat_near_zero_slope(self):
        result = analyze(_flat(60))
        assert result is not None
        assert abs(result.slope) < 0.1

    def test_slope_pct_annual_for_uptrend(self):
        result = analyze(_perfect_uptrend(60))
        assert result is not None
        assert result.slope_pct_annual > 0


class TestRSquared:
    def test_r2_in_range(self):
        for bars in [_flat(60), _perfect_uptrend(60), _noisy_uptrend(60)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.r_squared <= 1.0

    def test_perfect_uptrend_high_r2(self):
        result = analyze(_perfect_uptrend(60))
        assert result is not None
        assert result.r_squared > 0.95

    def test_r2_label_valid(self):
        result = analyze(_flat(60))
        assert result is not None
        assert result.r_squared_label in {"strong", "moderate", "weak"}


class TestBands:
    def test_five_bands_returned(self):
        result = analyze(_flat(55))
        assert result is not None
        assert len(result.bands) == 5

    def test_bands_are_lrc_band(self):
        result = analyze(_flat(55))
        assert result is not None
        for b in result.bands:
            assert isinstance(b, LRCBand)

    def test_bands_sorted_by_sigma(self):
        result = analyze(_flat(55))
        assert result is not None
        sigmas = [b.sigma for b in result.bands]
        assert sigmas == sorted(sigmas)

    def test_upper_above_lower(self):
        result = analyze(_flat(55))
        assert result is not None
        upper = next(b for b in result.bands if b.sigma == 2.0)
        lower = next(b for b in result.bands if b.sigma == -2.0)
        assert upper.price >= lower.price


class TestResidual:
    def test_residual_z_near_zero_for_linear(self):
        result = analyze(_perfect_uptrend(60))
        assert result is not None
        assert abs(result.residual_z) < 0.5

    def test_position_label_valid(self):
        for bars in [_flat(55), _perfect_uptrend(60), _noisy_uptrend(60)]:
            result = analyze(bars)
            if result:
                assert result.position_label in {
                    "above_upper", "upper_half", "on_line", "lower_half", "below_lower"
                }


class TestTrend:
    def test_trend_direction_valid(self):
        for bars in [_flat(55), _perfect_uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.trend_direction in {"up", "down", "flat"}

    def test_uptrend_detected(self):
        result = analyze(_perfect_uptrend(60))
        assert result is not None
        assert result.trend_direction == "up"

    def test_downtrend_detected(self):
        result = analyze(_downtrend(60))
        assert result is not None
        assert result.trend_direction == "down"


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_symbol_stored(self):
        result = analyze(_flat(55), symbol="COST")
        assert result is not None
        assert result.symbol == "COST"

    def test_current_price_correct(self):
        result = analyze(_flat(55, price=150.0))
        assert result is not None
        assert abs(result.current_price - 150.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(55))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_linreg(self):
        result = analyze(_flat(55))
        assert result is not None
        text = result.verdict.lower()
        assert "linreg" in text or "regression" in text or "slope" in text
