"""Tests for Ichimoku Cloud feature."""

from __future__ import annotations

from amms.data.bars import Bar
from amms.features.ichimoku import IchimokuResult, ichimoku


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _steady_bars(n: int, base: float = 100.0, step: float = 0.0) -> list[Bar]:
    return [_bar(base + i * step + 1.0, base + i * step - 1.0, base + i * step, i) for i in range(n)]


class TestIchimoku:
    def test_returns_none_for_insufficient_bars(self) -> None:
        bars = _steady_bars(30)
        assert ichimoku(bars) is None

    def test_returns_result_for_52_plus_bars(self) -> None:
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert isinstance(result, IchimokuResult)

    def test_position_valid_value(self) -> None:
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert result.position in ("above_cloud", "below_cloud", "in_cloud")

    def test_momentum_valid_value(self) -> None:
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert result.momentum in ("bullish", "bearish", "neutral")

    def test_cloud_color_valid(self) -> None:
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert result.cloud_color in ("green", "red", "flat")

    def test_cloud_top_gte_cloud_bottom(self) -> None:
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert result.cloud_top >= result.cloud_bottom

    def test_price_matches_last_close(self) -> None:
        bars = _steady_bars(60, base=150.0)
        result = ichimoku(bars)
        assert result is not None
        assert abs(result.price - bars[-1].close) < 0.01

    def test_uptrend_bullish_momentum(self) -> None:
        """Strongly trending up: Tenkan (short) > Kijun (long)."""
        bars = _steady_bars(60, step=2.0)
        result = ichimoku(bars)
        assert result is not None
        assert result.momentum == "bullish"

    def test_above_cloud_for_strong_uptrend(self) -> None:
        """Price far above any cloud level → above_cloud."""
        bars = _steady_bars(60, step=5.0)
        result = ichimoku(bars)
        assert result is not None
        assert result.position == "above_cloud"

    def test_tenkan_and_kijun_are_midpoints(self) -> None:
        """For flat bars, tenkan = kijun = midpoint."""
        bars = _steady_bars(60)
        result = ichimoku(bars)
        assert result is not None
        assert abs(result.tenkan - result.kijun) < 0.01
