"""Tests for amms.analysis.vortex."""

from __future__ import annotations

import pytest

from amms.analysis.vortex import VortexReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


MIN_BARS = 30  # 14 + 10 + 3 + margin


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price) for _ in range(n)]


def _uptrend(n: int = 80, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price += step
    return bars


def _downtrend(n: int = 80, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price = max(price - step, 1.0)
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, VortexReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = low = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestComponents:
    def test_vi_plus_positive(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.vi_plus >= 0

    def test_vi_minus_positive(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.vi_minus >= 0

    def test_spread_is_vip_minus_vim(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.vortex_spread - (result.vi_plus - result.vi_minus)) < 1e-4

    def test_bullish_consistent_with_vip_vim(self):
        result = analyze(_uptrend())
        assert result is not None
        if result.vi_plus > result.vi_minus:
            assert result.bullish is True
        elif result.vi_plus < result.vi_minus:
            assert result.bullish is False

    def test_period_stored(self):
        result = analyze(_flat(MIN_BARS), period=14)
        assert result is not None
        assert result.period == 14


class TestUptrend:
    def test_uptrend_vi_plus_dominant(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.vi_plus >= result.vi_minus

    def test_uptrend_bullish(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.bullish is True


class TestDowntrend:
    def test_downtrend_vi_minus_dominant(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.vi_minus >= result.vi_plus

    def test_downtrend_not_bullish(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.bullish is False


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_uptrend_positive_score(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.score >= 0


class TestCross:
    def test_cross_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.cross_bullish, bool)
        assert isinstance(result.cross_bearish, bool)

    def test_not_both_cross(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.cross_bullish and result.cross_bearish)


class TestSeries:
    def test_vi_plus_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.vi_plus_series) > 0

    def test_vi_minus_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.vi_minus_series) > 0

    def test_vi_plus_series_last_matches(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.vi_plus_series[-1] - result.vi_plus) < 1e-4


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="QQQ")
        assert result is not None
        assert result.symbol == "QQQ"

    def test_bars_used_correct(self):
        bars = _flat(50)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 50


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_vortex(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Vortex" in result.verdict or "VI" in result.verdict
