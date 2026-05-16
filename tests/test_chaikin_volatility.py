"""Tests for amms.analysis.chaikin_volatility."""

from __future__ import annotations

import pytest

from amms.analysis.chaikin_volatility import CVReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float = None):
        self.high = high
        self.low = low
        self.close = close if close is not None else (high + low) / 2


def _wide_bars(n: int = 80, spread: float = 5.0) -> list[_Bar]:
    """Bars with large HL range — high volatility."""
    return [_Bar(100.0 + spread, 100.0 - spread) for _ in range(n)]


def _narrow_bars(n: int = 80, spread: float = 0.5) -> list[_Bar]:
    """Bars with small HL range — low volatility."""
    return [_Bar(100.0 + spread, 100.0 - spread) for _ in range(n)]


def _expanding_bars(n: int = 80) -> list[_Bar]:
    """Gradually widening bars — expanding volatility."""
    bars = []
    for i in range(n):
        spread = 0.5 + i * 0.1
        bars.append(_Bar(100.0 + spread, 100.0 - spread))
    return bars


def _contracting_bars(n: int = 80) -> list[_Bar]:
    """Gradually narrowing bars — contracting volatility."""
    bars = []
    for i in range(n):
        spread = max(0.1, 5.0 - i * 0.06)
        bars.append(_Bar(100.0 + spread, 100.0 - spread))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_wide_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert isinstance(result, CVReport)

    def test_no_high_attr_returns_none(self):
        class _Bad:
            low = 1.0
        assert analyze([_Bad()] * 80) is None

    def test_inverted_bars_returns_none(self):
        # high < low is invalid
        class _Inv:
            high = 90.0
            low = 110.0
        assert analyze([_Inv()] * 80) is None


class TestCVValue:
    def test_expanding_cv_positive(self):
        result = analyze(_expanding_bars(80))
        assert result is not None
        assert result.cv > 0

    def test_contracting_cv_negative(self):
        result = analyze(_contracting_bars(80))
        assert result is not None
        assert result.cv < 0

    def test_constant_bars_cv_near_zero(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert abs(result.cv) < 0.01

    def test_cv_is_float(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert isinstance(result.cv, float)


class TestFlags:
    def test_expanding_flag_set(self):
        result = analyze(_expanding_bars(80))
        assert result is not None
        assert result.expanding is True

    def test_contracting_flag_set(self):
        result = analyze(_contracting_bars(80))
        assert result is not None
        assert result.contracting is True

    def test_not_both_expanding_contracting(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert not (result.expanding and result.contracting)

    def test_spike_is_bool(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert isinstance(result.spike, bool)


class TestPercentile:
    def test_percentile_in_range(self):
        for bars in [_expanding_bars(80), _contracting_bars(80), _wide_bars(80)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.cv_percentile <= 100.0

    def test_avg_cv_is_float(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert isinstance(result.avg_cv, float)

    def test_cv_std_non_negative(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert result.cv_std >= 0.0


class TestScore:
    def test_score_in_range(self):
        for bars in [_expanding_bars(80), _contracting_bars(80), _wide_bars(80)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_expanding_positive_score(self):
        result = analyze(_expanding_bars(80))
        assert result is not None
        assert result.score > 0

    def test_contracting_negative_or_zero_score(self):
        result = analyze(_contracting_bars(80))
        assert result is not None
        assert result.score <= 0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"expanding", "contracting", "neutral", "spike", "squeeze"}
        for bars in [_expanding_bars(80), _contracting_bars(80), _wide_bars(80)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_expanding_signal(self):
        result = analyze(_expanding_bars(80))
        assert result is not None
        assert result.signal in {"expanding", "spike", "neutral"}

    def test_contracting_signal(self):
        result = analyze(_contracting_bars(80))
        assert result is not None
        assert result.signal in {"contracting", "squeeze", "neutral"}


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert len(result.history) == 30

    def test_history_length_custom(self):
        result = analyze(_wide_bars(80), history=15)
        assert result is not None
        assert len(result.history) == 15

    def test_last_history_matches_cv(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert result.history[-1] == result.cv


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_wide_bars(80), symbol="VIX")
        assert result is not None
        assert result.symbol == "VIX"

    def test_bars_used_correct(self):
        bars = _wide_bars(90)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 90

    def test_ema_hl_positive(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert result.ema_hl > 0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_cv(self):
        result = analyze(_wide_bars(80))
        assert result is not None
        assert "CV" in result.verdict or "Chaikin" in result.verdict
