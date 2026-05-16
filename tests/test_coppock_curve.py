"""Tests for amms.analysis.coppock_curve."""

from __future__ import annotations

import pytest

from amms.analysis.coppock_curve import CoppockReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


# Short mode: roc1=14, roc2=11, wma=10 → min = max(14,11) + 10 + 20 + 5 = 49
MIN_BARS_SHORT = 55


def _flat(n: int = MIN_BARS_SHORT, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = MIN_BARS_SHORT, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _downtrend(n: int = MIN_BARS_SHORT, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        # Short mode requires ~55 bars
        assert analyze(_flat(10), short_mode=True) is None

    def test_returns_result_short_mode(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result, CoppockReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS_SHORT, short_mode=True) is None


class TestCoppockValues:
    def test_coppock_is_float(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.coppock, float)

    def test_coppock_prev_is_float(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.coppock_prev, float)

    def test_rising_bool(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.rising, bool)
        # Consistent with values
        assert result.rising == (result.coppock > result.coppock_prev)

    def test_above_zero_bool(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.above_zero, bool)
        assert result.above_zero == (result.coppock > 0)

    def test_uptrend_coppock_positive(self):
        result = analyze(_uptrend(), short_mode=True)
        assert result is not None
        assert result.coppock > 0

    def test_downtrend_coppock_negative(self):
        result = analyze(_downtrend(), short_mode=True)
        assert result is not None
        assert result.coppock < 0


class TestSignals:
    def test_buy_sell_bools(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.buy_signal, bool)
        assert isinstance(result.sell_alert, bool)

    def test_buy_signal_only_when_below_zero_and_rising(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        if result.buy_signal:
            assert result.rising and not result.above_zero

    def test_sell_alert_only_when_above_zero_and_falling(self):
        result = analyze(_uptrend(), short_mode=True)
        assert result is not None
        if result.sell_alert:
            assert not result.rising and result.above_zero

    def test_signal_valid(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars, mode in [(_flat(MIN_BARS_SHORT), True), (_uptrend(), True), (_downtrend(), True)]:
            result = analyze(bars, short_mode=mode)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS_SHORT), _uptrend(), _downtrend()]:
            result = analyze(bars, short_mode=True)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestAcceleration:
    def test_acceleration_is_float(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert isinstance(result.acceleration, float)


class TestSeries:
    def test_coppock_series_non_empty(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert len(result.coppock_series) > 0

    def test_coppock_series_last_matches(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert abs(result.coppock_series[-1] - result.coppock) < 1e-4


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS_SHORT), symbol="SPY", short_mode=True)
        assert result is not None
        assert result.symbol == "SPY"

    def test_bars_used_correct(self):
        bars = _flat(MIN_BARS_SHORT + 5)
        result = analyze(bars, short_mode=True)
        assert result is not None
        assert result.bars_used == MIN_BARS_SHORT + 5


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_coppock(self):
        result = analyze(_flat(MIN_BARS_SHORT), short_mode=True)
        assert result is not None
        assert "Coppock" in result.verdict
