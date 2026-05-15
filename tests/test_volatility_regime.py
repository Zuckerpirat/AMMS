"""Tests for volatility regime classifier."""

from __future__ import annotations

from amms.analysis.volatility_regime import VolatilityRegime, classify
from amms.data.bars import Bar


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _stable_bars(n: int = 50) -> list[Bar]:
    return [_bar(101.0, 99.0, 100.0, i) for i in range(n)]


def _volatile_bars(n: int = 50) -> list[Bar]:
    bars = []
    for i in range(n):
        spread = 1.0 + (i / n) * 10.0  # increasing volatility
        bars.append(_bar(100.0 + spread, 100.0 - spread, 100.0, i))
    return bars


class TestVolatilityRegime:
    def test_returns_none_for_insufficient_bars(self) -> None:
        bars = _stable_bars(20)
        assert classify(bars, atr_period=14) is None

    def test_returns_result_for_enough_bars(self) -> None:
        result = classify(_stable_bars(50))
        assert result is not None
        assert isinstance(result, VolatilityRegime)

    def test_regime_valid_value(self) -> None:
        result = classify(_stable_bars(50))
        assert result is not None
        assert result.regime in ("low", "normal", "high", "extreme")

    def test_percentile_in_range(self) -> None:
        result = classify(_stable_bars(50))
        assert result is not None
        assert 0.0 <= result.percentile <= 100.0

    def test_size_multiplier_in_range(self) -> None:
        result = classify(_stable_bars(50))
        assert result is not None
        assert 0.0 < result.recommended_size_mult <= 1.0

    def test_stable_bars_low_regime(self) -> None:
        # Stable bars → low volatility regime
        result = classify(_stable_bars(50))
        assert result is not None
        assert result.regime in ("low", "normal")

    def test_increasing_vol_high_regime(self) -> None:
        # Bars with high recent spread compared to stable history
        bars = _stable_bars(40) + [_bar(115.0, 85.0, 100.0, i + 40) for i in range(10)]
        result = classify(bars)
        assert result is not None
        assert result.regime in ("high", "extreme")

    def test_atr_pct_non_negative(self) -> None:
        result = classify(_stable_bars(50))
        assert result is not None
        assert result.atr_pct_of_price >= 0.0

    def test_extreme_regime_small_size_mult(self) -> None:
        bars = _stable_bars(40) + [_bar(120.0 + i * 5, 80.0 - i * 5, 100.0, i + 40) for i in range(10)]
        result = classify(bars)
        assert result is not None
        assert result.recommended_size_mult <= 0.6
