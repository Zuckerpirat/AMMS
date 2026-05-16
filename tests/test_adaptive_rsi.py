"""Tests for amms.analysis.adaptive_rsi."""

from __future__ import annotations

import pytest

from amms.analysis.adaptive_rsi import ARSIReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close
        self.high = close + 0.5
        self.low = close - 0.5
        self.open = close


def _up_bars(n: int = 100, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _down_bars(n: int = 100, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(1.0, start - i * step)) for i in range(n)]


def _flat_bars(n: int = 100, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price) for _ in range(n)]


def _oscillating_bars(n: int = 100) -> list[_Bar]:
    bars = []
    for i in range(n):
        bars.append(_Bar(100.0 + 5.0 * (1 if i % 6 < 3 else -1)))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_up_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert isinstance(result, ARSIReport)

    def test_no_close_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * 100) is None

    def test_zero_price_returns_none(self):
        assert analyze([_Bar(0.0)] * 100) is None


class TestARSIBounds:
    def test_arsi_in_range(self):
        for bars in [_up_bars(100), _down_bars(100), _flat_bars(100)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.arsi <= 100.0

    def test_arsi_signal_in_range(self):
        for bars in [_up_bars(100), _down_bars(100)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.arsi_signal <= 100.0

    def test_score_in_range(self):
        for bars in [_up_bars(100), _down_bars(100), _flat_bars(100)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_er_in_range(self):
        for bars in [_up_bars(100), _down_bars(100), _oscillating_bars(100)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.er <= 1.0


class TestEffectivePeriod:
    def test_effective_period_in_bounds(self):
        for bars in [_up_bars(100), _flat_bars(100), _oscillating_bars(100)]:
            result = analyze(bars)
            if result:
                assert 2 <= result.effective_period <= 30

    def test_trending_period_lower_than_choppy(self):
        # High ER → shorter period; low ER (choppy) → longer period
        result_trend = analyze(_up_bars(100))
        result_chop  = analyze(_oscillating_bars(100))
        if result_trend and result_chop:
            # Not guaranteed but generally holds for strong trend vs choppy
            assert result_trend.effective_period <= result_chop.effective_period


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_up_bars(100), _down_bars(100), _flat_bars(100), _oscillating_bars(100)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_uptrend_bullish_score(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert result.score > 0

    def test_downtrend_bearish_score(self):
        result = analyze(_down_bars(100))
        assert result is not None
        assert result.score < 0


class TestFlags:
    def test_overbought_consistent(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert result.overbought == (result.arsi > 70.0)

    def test_oversold_consistent(self):
        result = analyze(_down_bars(100))
        assert result is not None
        assert result.oversold == (result.arsi < 30.0)

    def test_bullish_consistent(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert result.bullish == (result.arsi > result.arsi_signal)

    def test_cross_up_is_bool(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert isinstance(result.cross_up, bool)

    def test_cross_down_is_bool(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert isinstance(result.cross_down, bool)

    def test_not_both_cross(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert not (result.cross_up and result.cross_down)


class TestDivergence:
    def test_price_direction_valid(self):
        for bars in [_up_bars(100), _down_bars(100), _flat_bars(100)]:
            result = analyze(bars)
            if result:
                assert result.price_direction in {"up", "down", "flat"}

    def test_arsi_direction_valid(self):
        for bars in [_up_bars(100), _down_bars(100), _flat_bars(100)]:
            result = analyze(bars)
            if result:
                assert result.arsi_direction in {"up", "down", "flat"}

    def test_divergence_is_bool(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert isinstance(result.divergence, bool)


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert len(result.history) == 20

    def test_history_length_custom(self):
        result = analyze(_up_bars(100), history=10)
        assert result is not None
        assert len(result.history) == 10

    def test_history_in_range(self):
        result = analyze(_up_bars(100))
        assert result is not None
        for v in result.history:
            assert 0.0 <= v <= 100.0

    def test_last_history_matches_arsi(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert result.history[-1] == result.arsi


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(100), symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_bars_used_correct(self):
        bars = _up_bars(110)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 110


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_arsi(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert "ARSI" in result.verdict

    def test_verdict_mentions_er(self):
        result = analyze(_up_bars(100))
        assert result is not None
        assert "ER=" in result.verdict
