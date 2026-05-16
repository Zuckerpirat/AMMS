"""Tests for amms.analysis.natr."""

from __future__ import annotations

import pytest

from amms.analysis.natr import NATRReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high = high
        self.low = low
        self.close = close
        self.open = (high + low) / 2


def _volatile_bars(n: int = 80, spread: float = 5.0, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + spread, price - spread, price) for _ in range(n)]


def _calm_bars(n: int = 80, spread: float = 0.5, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + spread, price - spread, price) for _ in range(n)]


def _rising_bars(n: int = 80) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = 100.0 + i
        bars.append(_Bar(price + 2.0, price - 2.0, price))
    return bars


def _mixed_volatility(n: int = 80) -> list[_Bar]:
    """Low volatility first half, high second half."""
    bars = []
    for i in range(n):
        spread = 0.5 if i < n // 2 else 5.0
        bars.append(_Bar(100.0 + spread, 100.0 - spread, 100.0))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_volatile_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert isinstance(result, NATRReport)

    def test_no_high_attr_returns_none(self):
        class _Bad:
            low = close = 100.0
        assert analyze([_Bad()] * 80) is None

    def test_zero_close_returns_none(self):
        assert analyze([_Bar(1.0, 0.0, 0.0)] * 80) is None


class TestNATRValue:
    def test_natr_positive(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.natr > 0

    def test_natr_percentage_reasonable(self):
        # 5 spread on 100 price ≈ 5% NATR (ATR ≈ spread)
        result = analyze(_volatile_bars(80, spread=5.0, price=100.0))
        assert result is not None
        assert 2.0 < result.natr <= 10.0  # spread=5 → HL=10 → NATR≈10%

    def test_calm_natr_smaller(self):
        vol   = analyze(_volatile_bars(80, spread=5.0))
        calm  = analyze(_calm_bars(80, spread=0.5))
        if vol and calm:
            assert calm.natr < vol.natr

    def test_atr_positive(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.atr > 0

    def test_natr_fast_slow_exist(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.natr_fast > 0
        assert result.natr_slow > 0


class TestRegime:
    def test_regime_valid_values(self):
        for bars in [_volatile_bars(80), _calm_bars(80), _rising_bars(80)]:
            result = analyze(bars)
            if result:
                assert result.regime in {"low", "normal", "elevated", "high"}

    def test_signal_valid_values(self):
        valid = {"calm", "normal", "active", "volatile", "extreme"}
        for bars in [_volatile_bars(80), _calm_bars(80), _rising_bars(80)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid


class TestFlags:
    def test_slope_up_is_bool(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert isinstance(result.slope_up, bool)

    def test_compression_is_bool(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert isinstance(result.compression, bool)

    def test_compression_fast_less_slow(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.compression == (result.natr_fast < result.natr_slow)


class TestPercentile:
    def test_percentile_in_range(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert 0.0 <= result.natr_percentile <= 100.0

    def test_avg_natr_positive(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.avg_natr > 0

    def test_std_non_negative(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.natr_std >= 0


class TestScore:
    def test_score_in_range(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_high_vol_high_score(self):
        result = analyze(_mixed_volatility(80))
        assert result is not None
        # After switching to high volatility, score should be elevated
        assert result.score > 50


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert len(result.history) == 30

    def test_history_length_custom(self):
        result = analyze(_volatile_bars(80), history=15)
        assert result is not None
        assert len(result.history) == 15

    def test_last_history_matches_natr(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert result.history[-1] == result.natr

    def test_history_all_positive(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        for v in result.history:
            assert v >= 0


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_volatile_bars(80), symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_bars_used_correct(self):
        bars = _volatile_bars(90)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 90


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_natr(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert "NATR" in result.verdict

    def test_verdict_mentions_atr(self):
        result = analyze(_volatile_bars(80))
        assert result is not None
        assert "ATR" in result.verdict
