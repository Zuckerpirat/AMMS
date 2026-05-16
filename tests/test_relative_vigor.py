"""Tests for amms.analysis.relative_vigor."""

from __future__ import annotations

import pytest

from amms.analysis.relative_vigor import RVIReport, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float):
        self.open  = open_
        self.high  = high
        self.low   = low
        self.close = close


# min_bars = 10 + 4 + 4 + 15 + 5 = 38
MIN_BARS = 45


def _up_bars(n: int = MIN_BARS, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    """Closes > Opens (bullish vigor)."""
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = price + step
        bars.append(_Bar(o, c + 0.5, o - 0.2, c))
        price = c
    return bars


def _down_bars(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    """Closes < Opens (bearish vigor)."""
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = max(price - step, 1.0)
        bars.append(_Bar(o, o + 0.2, c - 0.5, c))
        price = c
    return bars


def _flat_bars(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    """Close == Open (neutral vigor)."""
    return [_Bar(price, price + 1.0, price - 1.0, price) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat_bars(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert isinstance(result, RVIReport)

    def test_no_open_attr_returns_none(self):
        class _NoOpen:
            high = low = close = 100.0
        assert analyze([_NoOpen()] * MIN_BARS) is None


class TestRVI:
    def test_rvi_is_float(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert isinstance(result.rvi, float)

    def test_signal_is_float(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert isinstance(result.rvi_signal, float)

    def test_histogram_is_rvi_minus_signal(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert abs(result.rvi_histogram - (result.rvi - result.rvi_signal)) < 1e-6

    def test_up_bars_rvi_positive(self):
        result = analyze(_up_bars())
        assert result is not None
        assert result.rvi > 0

    def test_down_bars_rvi_negative(self):
        result = analyze(_down_bars())
        assert result is not None
        assert result.rvi < 0

    def test_bullish_consistent(self):
        for bars in [_flat_bars(MIN_BARS), _up_bars(), _down_bars()]:
            result = analyze(bars)
            if result:
                assert result.bullish == (result.rvi > 0)

    def test_period_stored(self):
        result = analyze(_flat_bars(MIN_BARS), period=10)
        assert result is not None
        assert result.period == 10


class TestCross:
    def test_cross_bools(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert isinstance(result.cross_up, bool)
        assert isinstance(result.cross_down, bool)

    def test_not_both_cross(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert not (result.cross_up and result.cross_down)


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat_bars(MIN_BARS), _up_bars(), _down_bars()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat_bars(MIN_BARS), _up_bars(), _down_bars()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_up_bars_bullish_signal(self):
        result = analyze(_up_bars())
        assert result is not None
        assert result.signal in {"bull", "strong_bull", "neutral"}

    def test_down_bars_bearish_signal(self):
        result = analyze(_down_bars())
        assert result is not None
        assert result.signal in {"bear", "strong_bear", "neutral"}


class TestSeries:
    def test_rvi_series_non_empty(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert len(result.rvi_series) > 0

    def test_signal_series_non_empty(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert len(result.signal_series) > 0

    def test_rvi_series_last_matches(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert abs(result.rvi_series[-1] - result.rvi) < 1e-5


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat_bars(MIN_BARS), symbol="GOOG")
        assert result is not None
        assert result.symbol == "GOOG"

    def test_bars_used_correct(self):
        bars = _flat_bars(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_rvi(self):
        result = analyze(_flat_bars(MIN_BARS))
        assert result is not None
        assert "RVI" in result.verdict
