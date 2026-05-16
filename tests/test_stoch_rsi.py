"""Tests for amms.analysis.stoch_rsi."""

from __future__ import annotations

import pytest

from amms.analysis.stoch_rsi import StochRSIReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close
        self.high = close + 0.5
        self.low = close - 0.5
        self.open = close


def _up_bars(n: int = 80, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _down_bars(n: int = 80, start: float = 180.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(1.0, start - i * step)) for i in range(n)]


def _flat_bars(n: int = 80, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price) for _ in range(n)]


def _oscillating_bars(n: int = 80) -> list[_Bar]:
    bars = []
    for i in range(n):
        bars.append(_Bar(100.0 + 10.0 * (1 if i % 10 < 5 else -1)))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_up_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert isinstance(result, StochRSIReport)

    def test_no_close_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * 80) is None

    def test_zero_price_returns_none(self):
        assert analyze([_Bar(0.0)] * 80) is None


class TestBounds:
    def test_k_in_range(self):
        for bars in [_up_bars(80), _down_bars(80), _flat_bars(80)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.k <= 100.0

    def test_d_in_range(self):
        for bars in [_up_bars(80), _down_bars(80)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.d <= 100.0

    def test_stoch_rsi_in_range(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert 0.0 <= result.stoch_rsi <= 100.0

    def test_rsi_in_range(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert 0.0 <= result.rsi <= 100.0

    def test_score_in_range(self):
        for bars in [_up_bars(80), _down_bars(80), _flat_bars(80)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_up_bars(80), _down_bars(80), _flat_bars(80), _oscillating_bars(80)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_uptrend_overbought(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.overbought == (result.k > 80.0)

    def test_downtrend_oversold(self):
        result = analyze(_down_bars(80))
        assert result is not None
        assert result.oversold == (result.k < 20.0)


class TestFlags:
    def test_overbought_consistent(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.overbought == (result.k > 80.0)

    def test_oversold_consistent(self):
        result = analyze(_down_bars(80))
        assert result is not None
        assert result.oversold == (result.k < 20.0)

    def test_k_above_d_consistent(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.k_above_d == (result.k > result.d)

    def test_not_both_cross(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert not (result.cross_up and result.cross_down)

    def test_cross_flags_are_bool(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert isinstance(result.cross_up, bool)
        assert isinstance(result.cross_down, bool)


class TestHistory:
    def test_history_k_length(self):
        result = analyze(_up_bars(80), history=10)
        assert result is not None
        assert len(result.history_k) == 10

    def test_history_d_length(self):
        result = analyze(_up_bars(80), history=10)
        assert result is not None
        assert len(result.history_d) == 10

    def test_history_k_in_range(self):
        result = analyze(_up_bars(80))
        assert result is not None
        for v in result.history_k:
            assert 0.0 <= v <= 100.0

    def test_last_history_k_matches_k(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.history_k[-1] == result.k

    def test_last_history_d_matches_d(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.history_d[-1] == result.d


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(80), symbol="AMZN")
        assert result is not None
        assert result.symbol == "AMZN"

    def test_bars_used_correct(self):
        bars = _up_bars(90)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 90


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_stochrsi(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert "StochRSI" in result.verdict

    def test_verdict_mentions_k_and_d(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert "%K=" in result.verdict and "%D=" in result.verdict
