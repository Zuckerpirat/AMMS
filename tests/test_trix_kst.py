"""Tests for amms.analysis.trix_kst."""

from __future__ import annotations

import pytest

from amms.analysis.trix_kst import TrixKSTReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


# min_bars = 3*18 + 9 + 15 + 15 + 5 = 54 + 44 = ~98
MIN_BARS = 110


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(30)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, TrixKSTReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None


class TestTrix:
    def test_trix_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.trix, float)

    def test_trix_signal_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.trix_signal, float)

    def test_trix_histogram_is_trix_minus_signal(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.trix_histogram - (result.trix - result.trix_signal)) < 1e-6

    def test_trix_bullish_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.trix_bullish, bool)

    def test_uptrend_trix_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.trix > 0

    def test_downtrend_trix_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.trix < 0

    def test_cross_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.trix_cross_up, bool)
        assert isinstance(result.trix_cross_down, bool)
        assert not (result.trix_cross_up and result.trix_cross_down)


class TestKST:
    def test_kst_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kst, float)

    def test_kst_signal_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kst_signal, float)

    def test_kst_histogram_is_kst_minus_signal(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.kst_histogram - (result.kst - result.kst_signal)) < 1e-4

    def test_uptrend_kst_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.kst > 0

    def test_downtrend_kst_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.kst < 0

    def test_kst_cross_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.kst_cross_up, bool)
        assert isinstance(result.kst_cross_down, bool)
        assert not (result.kst_cross_up and result.kst_cross_down)


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


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="SPY")
        assert result is not None
        assert result.symbol == "SPY"

    def test_bars_used_correct(self):
        bars = _flat(MIN_BARS)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == MIN_BARS

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=420.0))
        assert result is not None
        assert abs(result.current_price - 420.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_trix(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "TRIX" in result.verdict

    def test_verdict_mentions_kst(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "KST" in result.verdict
