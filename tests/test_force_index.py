"""Tests for amms.analysis.force_index."""

from __future__ import annotations

import pytest

from amms.analysis.force_index import ForceIndexReport, analyze


class _Bar:
    def __init__(self, close: float, volume: float = 1000.0):
        self.close  = close
        self.volume = volume


# min_bars = 13 + 20 + 15 + 5 = 53
MIN_BARS = 60


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, 1000.0) for _ in range(n)]


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step, 2000.0) for i in range(n)]


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0), 2000.0) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, ForceIndexReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None

    def test_no_volume_still_works(self):
        class _NoVol:
            close = 100.0
        result = analyze([_NoVol()] * MIN_BARS)
        assert result is not None


class TestForceIndex:
    def test_raw_fi_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.raw_fi, float)

    def test_fi2_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.fi_2, float)

    def test_fi13_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.fi_13, float)

    def test_fi2_positive_consistent(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.fi2_positive == (result.fi_2 > 0)

    def test_fi13_positive_consistent(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.fi13_positive == (result.fi_13 > 0)

    def test_uptrend_fi13_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.fi_13 > 0

    def test_downtrend_fi13_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.fi_13 < 0


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


class TestSetups:
    def test_buy_sell_setup_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.buy_setup, bool)
        assert isinstance(result.sell_setup, bool)

    def test_not_both_buy_and_sell(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.buy_setup and result.sell_setup)

    def test_trend_confirmed_valid(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.trend_confirmed in {"up", "down", "unclear"}

    def test_zero_cross_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.fi13_zero_cross, bool)

    def test_spike_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.fi2_spike, bool)


class TestSeries:
    def test_fi2_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.fi2_series) > 0

    def test_fi13_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.fi13_series) > 0


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="MSFT")
        assert result is not None
        assert result.symbol == "MSFT"

    def test_bars_used_correct(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 70


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_force(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Force" in result.verdict or "FI" in result.verdict
