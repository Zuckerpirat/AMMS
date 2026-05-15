"""Tests for Z-score feature."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.zscore import zscore, zscore_series


def _bar(close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, close + 0.5, close - 0.5, close, 1000.0)


def _bars(closes: list[float]) -> list[Bar]:
    return [_bar(c, i) for i, c in enumerate(closes)]


class TestZscore:
    def test_returns_none_if_insufficient_bars(self) -> None:
        bars = _bars([100.0] * 10)
        assert zscore(bars, n=20) is None

    def test_returns_zero_for_uniform_prices(self) -> None:
        bars = _bars([100.0] * 20)
        assert zscore(bars, n=20) == 0.0

    def test_positive_zscore_for_price_above_mean(self) -> None:
        closes = [100.0] * 19 + [110.0]
        result = zscore(_bars(closes), n=20)
        assert result is not None
        assert result > 0

    def test_negative_zscore_for_price_below_mean(self) -> None:
        closes = [100.0] * 19 + [90.0]
        result = zscore(_bars(closes), n=20)
        assert result is not None
        assert result < 0

    def test_extreme_positive_zscore(self) -> None:
        closes = [100.0] * 19 + [200.0]
        result = zscore(_bars(closes), n=20)
        assert result is not None
        assert result > 2.0

    def test_extreme_negative_zscore(self) -> None:
        closes = [100.0] * 19 + [0.0]
        result = zscore(_bars(closes), n=20)
        assert result is not None
        assert result < -2.0

    def test_raises_for_n_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            zscore(_bars([100.0] * 20), n=1)

    def test_uses_last_n_bars_only(self) -> None:
        closes_base = [100.0] * 20
        closes_extra = [50.0] * 5 + closes_base
        r1 = zscore(_bars(closes_base), n=20)
        r2 = zscore(_bars(closes_extra), n=20)
        assert r1 == r2


class TestZscoreSeries:
    def test_empty_series_for_too_few_bars(self) -> None:
        assert zscore_series(_bars([100.0] * 10), n=20) == []

    def test_series_length(self) -> None:
        bars = _bars([float(100 + i) for i in range(30)])
        series = zscore_series(bars, n=20)
        assert len(series) == 30 - 20 + 1

    def test_series_contains_floats(self) -> None:
        bars = _bars([float(100 + (i % 5)) for i in range(25)])
        series = zscore_series(bars, n=20)
        assert all(isinstance(v, float) for v in series)

    def test_raises_for_n_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            zscore_series(_bars([100.0] * 20), n=1)

    def test_zero_std_produces_zero_entry(self) -> None:
        bars = _bars([100.0] * 25)
        series = zscore_series(bars, n=20)
        assert all(v == 0.0 for v in series)
