"""Tests for amms.analysis.ultimate_oscillator."""

from __future__ import annotations

import pytest

from amms.analysis.ultimate_oscillator import UOReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high = high
        self.low = low
        self.close = close
        self.open = (high + low) / 2


def _up_bars(n: int = 70, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = start + i * step
        bars.append(_Bar(price + 1.0, price - 0.5, price))
    return bars


def _down_bars(n: int = 70, start: float = 170.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = max(1.0, start - i * step)
        bars.append(_Bar(price + 0.5, price - 1.0, price))
    return bars


def _flat_bars(n: int = 70, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 0.5, price - 0.5, price) for _ in range(n)]


def _oscillating_bars(n: int = 70) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = 100.0 + 5.0 * (1 if i % 8 < 4 else -1)
        bars.append(_Bar(price + 1.0, price - 1.0, price))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_up_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert isinstance(result, UOReport)

    def test_no_high_attr_returns_none(self):
        class _Bad:
            low = close = 100.0
        assert analyze([_Bad()] * 70) is None

    def test_zero_close_returns_none(self):
        assert analyze([_Bar(1.0, 0.5, 0.0)] * 70) is None


class TestUOBounds:
    def test_uo_in_range(self):
        for bars in [_up_bars(70), _down_bars(70), _flat_bars(70)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.uo <= 100.0

    def test_avg7_in_range(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert 0.0 <= result.avg7 <= 1.0

    def test_avg14_in_range(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert 0.0 <= result.avg14 <= 1.0

    def test_avg28_in_range(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert 0.0 <= result.avg28 <= 1.0

    def test_score_in_range(self):
        for bars in [_up_bars(70), _down_bars(70), _flat_bars(70)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_up_bars(70), _down_bars(70), _flat_bars(70), _oscillating_bars(70)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_uptrend_bullish_uo(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert result.uo >= 50

    def test_downtrend_bearish_uo(self):
        result = analyze(_down_bars(70))
        assert result is not None
        assert result.uo <= 50


class TestFlags:
    def test_overbought_consistent(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert result.overbought == (result.uo > 70.0)

    def test_oversold_consistent(self):
        result = analyze(_down_bars(70))
        assert result is not None
        assert result.oversold == (result.uo < 30.0)

    def test_slope_up_is_bool(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert isinstance(result.slope_up, bool)

    def test_divergence_are_bool(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert isinstance(result.bull_divergence, bool)
        assert isinstance(result.bear_divergence, bool)

    def test_not_both_divergences(self):
        for bars in [_up_bars(70), _down_bars(70), _flat_bars(70)]:
            result = analyze(bars)
            if result:
                assert not (result.bull_divergence and result.bear_divergence)


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert len(result.history) == 20

    def test_history_length_custom(self):
        result = analyze(_up_bars(70), history=10)
        assert result is not None
        assert len(result.history) == 10

    def test_history_in_range(self):
        result = analyze(_up_bars(70))
        assert result is not None
        for v in result.history:
            assert 0.0 <= v <= 100.0

    def test_last_history_matches_uo(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert result.history[-1] == result.uo


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(70), symbol="QQQ")
        assert result is not None
        assert result.symbol == "QQQ"

    def test_bars_used_correct(self):
        bars = _up_bars(80)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 80


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_uo(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert "UO" in result.verdict

    def test_verdict_mentions_averages(self):
        result = analyze(_up_bars(70))
        assert result is not None
        assert "Avg7" in result.verdict
