"""Tests for amms.analysis.chande_mo."""

from __future__ import annotations

import pytest

from amms.analysis.chande_mo import CMOReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


MIN_BARS = 20 + 9 + 20 + 5  # = 54


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, CMOReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None


class TestCMOValues:
    def test_cmo_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.cmo <= 100.0

    def test_abs_cmo_non_negative(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.abs_cmo >= 0

    def test_abs_cmo_equals_abs_cmo(self):
        result = analyze(_uptrend())
        assert result is not None
        assert abs(result.abs_cmo - abs(result.cmo)) < 0.01

    def test_uptrend_cmo_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.cmo > 0

    def test_downtrend_cmo_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.cmo < 0

    def test_period_stored(self):
        result = analyze(_flat(MIN_BARS), period=14)
        assert result is not None
        assert result.period == 14


class TestBooleanStates:
    def test_bullish_consistent_with_cmo(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.bullish == (result.cmo > 0)

    def test_overbought_consistent(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.overbought == (result.cmo > 50)

    def test_oversold_consistent(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.oversold == (result.cmo < -50)

    def test_not_both_overbought_and_oversold(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert not (result.overbought and result.oversold)

    def test_histogram_is_cmo_minus_signal(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.cmo_histogram - (result.cmo - result.cmo_signal)) < 0.01


class TestCross:
    def test_cross_bools_type(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        for attr in ["zero_cross_up", "zero_cross_down", "signal_cross_up", "signal_cross_down"]:
            assert isinstance(getattr(result, attr), bool)

    def test_not_both_zero_cross(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.zero_cross_up and result.zero_cross_down)

    def test_not_both_signal_cross(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.signal_cross_up and result.signal_cross_down)


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

    def test_uptrend_bullish_signal(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.signal in {"bull", "strong_bull", "neutral"}

    def test_downtrend_bearish_signal(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.signal in {"bear", "strong_bear", "neutral"}


class TestSeries:
    def test_cmo_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.cmo_series) > 0

    def test_signal_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.signal_series) > 0

    def test_cmo_series_last_matches(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.cmo_series[-1] - result.cmo) < 0.01


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_bars_used_correct(self):
        bars = _flat(MIN_BARS + 10)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == MIN_BARS + 10


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_cmo(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "CMO" in result.verdict
