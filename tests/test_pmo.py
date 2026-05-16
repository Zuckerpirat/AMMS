"""Tests for amms.analysis.pmo (Price Momentum Oscillator)."""

from __future__ import annotations

import pytest

from amms.analysis.pmo import PMOReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


MIN_BARS = 35 + 20 + 10 + 15 + 5  # = 85


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
        assert isinstance(result, PMOReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None


class TestPMO:
    def test_pmo_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.pmo, float)

    def test_pmo_signal_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.pmo_signal, float)

    def test_histogram_is_pmo_minus_signal(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.pmo_histogram - (result.pmo - result.pmo_signal)) < 1e-4

    def test_uptrend_pmo_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.pmo > 0

    def test_downtrend_pmo_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.pmo < 0

    def test_pmo_bullish_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.pmo_bullish, bool)

    def test_above_signal_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.above_signal, bool)


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


class TestOverboughtOversold:
    def test_pmo_pct_rank_in_range(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert 0.0 <= result.pmo_pct_rank <= 100.0

    def test_ob_os_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.overbought, bool)
        assert isinstance(result.oversold, bool)

    def test_not_both_ob_and_os(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.overbought and result.oversold)


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

    def test_uptrend_bullish_signal(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.signal in {"bull", "strong_bull", "neutral"}

    def test_downtrend_bearish_signal(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.signal in {"bear", "strong_bear", "neutral"}


class TestSeries:
    def test_pmo_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.pmo_series) > 0

    def test_signal_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.signal_series) > 0

    def test_pmo_series_last_matches(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.pmo_series[-1] - result.pmo) < 1e-4


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="AMZN")
        assert result is not None
        assert result.symbol == "AMZN"

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

    def test_verdict_mentions_pmo(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "PMO" in result.verdict
