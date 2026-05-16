"""Tests for amms.analysis.klinger_vo."""

from __future__ import annotations

import pytest

from amms.analysis.klinger_vo import KVOReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float, volume: float = 1000.0):
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume


MIN_BARS = 55 + 13 + 15 + 5  # slow + signal + history + margin = 88


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price, 1000.0) for _ in range(n)]


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price, 2000.0))
        price += step
    return bars


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price, 2000.0))
        price = max(price - step, 1.0)
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(30)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, KVOReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = low = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None

    def test_no_volume_still_works(self):
        class _NoVol:
            high = low = close = 100.0
        result = analyze([_NoVol()] * MIN_BARS)
        assert result is not None


class TestKVO:
    def test_kvo_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kvo, float)

    def test_signal_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kvo_signal, float)

    def test_histogram_is_kvo_minus_signal(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.kvo_histogram - (result.kvo - result.kvo_signal)) < 0.01

    def test_kvo_bullish_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kvo_bullish, bool)

    def test_kvo_bullish_consistent_with_value(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.kvo_bullish == (result.kvo > 0)

    def test_above_signal_consistent(self):
        for bars in [_flat(MIN_BARS), _uptrend()]:
            result = analyze(bars)
            if result:
                assert result.above_signal == (result.kvo > result.kvo_signal)


class TestSignal:
    def test_signal_valid(self):
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


class TestCross:
    def test_cross_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.cross_up, bool)
        assert isinstance(result.cross_down, bool)

    def test_not_both_cross(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.cross_up and result.cross_down)


class TestDivergence:
    def test_divergence_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.divergence, bool)

    def test_price_direction_valid(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.price_direction in {"up", "down", "flat"}

    def test_kvo_direction_valid(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.kvo_direction in {"up", "down", "flat"}

    def test_no_divergence_when_both_directions_match(self):
        result = analyze(_uptrend())
        assert result is not None
        if result.price_direction == result.kvo_direction and result.price_direction != "flat":
            assert not result.divergence


class TestSeries:
    def test_kvo_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.kvo_series) > 0

    def test_signal_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.signal_series) > 0

    def test_kvo_series_last_matches_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.kvo_series[-1] - result.kvo) < 0.1


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="GOOG")
        assert result is not None
        assert result.symbol == "GOOG"

    def test_bars_used_correct(self):
        bars = _flat(MIN_BARS + 5)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == MIN_BARS + 5

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=300.0))
        assert result is not None
        assert abs(result.current_price - 300.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_kvo(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "KVO" in result.verdict
