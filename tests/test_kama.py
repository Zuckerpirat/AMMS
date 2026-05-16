"""Tests for amms.analysis.kama (Kaufman's Adaptive Moving Average)."""

from __future__ import annotations

import pytest

from amms.analysis.kama import KAMAReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


MIN_BARS = 10 + 20 + 5  # period + history + margin = 35


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


def _noisy_flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    import math
    return [_Bar(price + math.sin(i * 0.5) * 2.0) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(5)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, KAMAReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None


class TestKAMA:
    def test_kama_positive(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.kama > 0

    def test_er_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.efficiency_ratio <= 1.0

    def test_sc_positive(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.smoothing_constant >= 0

    def test_uptrend_price_above_kama(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.price_above_kama is True

    def test_downtrend_price_below_kama(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.price_above_kama is False

    def test_price_above_consistent(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.price_above_kama == (result.current_price > result.kama)

    def test_uptrend_high_er(self):
        result = analyze(_uptrend(n=80))
        assert result is not None
        # Perfect uptrend should have high efficiency ratio
        assert result.efficiency_ratio > 0.5


class TestRegime:
    def test_er_regime_valid(self):
        valid = {"trending", "choppy", "transitional"}
        for bars in [_flat(MIN_BARS), _uptrend(), _noisy_flat()]:
            result = analyze(bars)
            if result:
                assert result.er_regime in valid

    def test_trending_regime_for_uptrend(self):
        result = analyze(_uptrend(n=80))
        assert result is not None
        # Sustained uptrend should be trending or transitional
        assert result.er_regime in {"trending", "transitional"}


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


class TestSeries:
    def test_kama_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.kama_series) > 0

    def test_er_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.er_series) > 0

    def test_kama_series_last_matches(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.kama_series[-1] - result.kama) < 0.01


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="AMZN")
        assert result is not None
        assert result.symbol == "AMZN"

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

    def test_verdict_mentions_kama(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "KAMA" in result.verdict
