"""Tests for Parabolic SAR."""

from __future__ import annotations

from amms.data.bars import Bar
from amms.features.parabolic_sar import SARResult, parabolic_sar


def _bar(high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, 1000.0)


def _uptrend(n: int = 20) -> list[Bar]:
    return [_bar(100.0 + i + 1, 99.0 + i, 100.0 + i, i) for i in range(n)]


def _downtrend(n: int = 20) -> list[Bar]:
    return [_bar(200.0 - i + 1, 199.0 - i, 200.0 - i, i) for i in range(n)]


class TestParabolicSAR:
    def test_returns_none_for_one_bar(self) -> None:
        assert parabolic_sar([_bar(102.0, 98.0, 100.0)]) is None

    def test_returns_result_for_two_bars(self) -> None:
        bars = [_bar(102.0, 98.0, 100.0, 0), _bar(103.0, 99.0, 101.0, 1)]
        result = parabolic_sar(bars)
        assert result is not None
        assert isinstance(result, SARResult)

    def test_uptrend_returns_up_trend(self) -> None:
        result = parabolic_sar(_uptrend(20))
        assert result is not None
        assert result.trend == "up"

    def test_downtrend_returns_down_trend(self) -> None:
        result = parabolic_sar(_downtrend(20))
        assert result is not None
        assert result.trend == "down"

    def test_sar_below_price_in_uptrend(self) -> None:
        bars = _uptrend(20)
        result = parabolic_sar(bars)
        assert result is not None
        price = bars[-1].close
        assert result.sar < price

    def test_sar_above_price_in_downtrend(self) -> None:
        bars = _downtrend(20)
        result = parabolic_sar(bars)
        assert result is not None
        price = bars[-1].close
        assert result.sar > price

    def test_acceleration_starts_at_initial(self) -> None:
        bars = [_bar(102.0, 98.0, 100.0, 0), _bar(103.0, 99.0, 101.0, 1)]
        result = parabolic_sar(bars, initial_af=0.02)
        assert result is not None
        assert result.acceleration >= 0.02

    def test_acceleration_does_not_exceed_max(self) -> None:
        result = parabolic_sar(_uptrend(50), max_af=0.20)
        assert result is not None
        assert result.acceleration <= 0.20

    def test_distance_pct_non_negative(self) -> None:
        result = parabolic_sar(_uptrend(20))
        assert result is not None
        assert result.distance_pct >= 0.0

    def test_trend_valid_values(self) -> None:
        result = parabolic_sar(_uptrend(20))
        assert result is not None
        assert result.trend in ("up", "down")

    def test_reversal_detected(self) -> None:
        """Build up then crash down — should detect downtrend reversal."""
        up = _uptrend(15)
        crash = [_bar(up[-1].high - j * 3, up[-1].low - j * 3, up[-1].close - j * 3, 15 + j) for j in range(10)]
        result = parabolic_sar(up + crash)
        assert result is not None
        assert result.trend == "down"
