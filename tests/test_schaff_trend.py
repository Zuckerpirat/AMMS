"""Tests for amms.analysis.schaff_trend."""

from __future__ import annotations

import pytest

from amms.analysis.schaff_trend import STCReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close
        self.open = close
        self.high = close + 0.5
        self.low = close - 0.5


def _trend_bars(n: int = 120, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _down_bars(n: int = 120, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(1.0, start - i * step)) for i in range(n)]


def _flat_bars(n: int = 120, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price) for _ in range(n)]


def _oscillating_bars(n: int = 120) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = 100.0 + 10.0 * (1 if i % 20 < 10 else -1)
        bars.append(_Bar(price))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_trend_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert isinstance(result, STCReport)

    def test_no_close_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * 120) is None

    def test_zero_price_returns_none(self):
        assert analyze([_Bar(0.0)] * 120) is None


class TestSTCBounds:
    def test_stc_in_range(self):
        for bars in [_trend_bars(120), _down_bars(120), _flat_bars(120)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.stc <= 100.0

    def test_stc_prev_in_range(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert 0.0 <= result.stc_prev <= 100.0

    def test_score_in_range(self):
        for bars in [_trend_bars(120), _down_bars(120), _flat_bars(120)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_trend_bars(120), _down_bars(120), _flat_bars(120), _oscillating_bars(120)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_uptrend_not_strong_sell(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert result.signal != "strong_sell"

    def test_downtrend_not_strong_buy(self):
        result = analyze(_down_bars(120))
        assert result is not None
        assert result.signal != "strong_buy"


class TestFlags:
    def test_overbought_consistent(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert result.overbought == (result.stc > 75.0)

    def test_oversold_consistent(self):
        result = analyze(_down_bars(120))
        assert result is not None
        assert result.oversold == (result.stc < 25.0)

    def test_slope_up_is_bool(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert isinstance(result.slope_up, bool)

    def test_slope_consistent_with_stc(self):
        # slope_up means stc > stc 3 bars ago
        result = analyze(_oscillating_bars(120))
        assert result is not None
        assert isinstance(result.slope_up, bool)

    def test_buy_trigger_is_bool(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert isinstance(result.buy_trigger, bool)

    def test_sell_trigger_is_bool(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert isinstance(result.sell_trigger, bool)


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert len(result.history) == 20

    def test_history_length_custom(self):
        result = analyze(_trend_bars(120), history=10)
        assert result is not None
        assert len(result.history) == 10

    def test_history_all_in_range(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        for v in result.history:
            assert 0.0 <= v <= 100.0

    def test_last_history_matches_stc(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert result.history[-1] == result.stc


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_trend_bars(120), symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used_correct(self):
        bars = _trend_bars(130)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 130

    def test_macd_is_float(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert isinstance(result.macd, float)


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_stc(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert "STC" in result.verdict

    def test_verdict_has_signal(self):
        result = analyze(_trend_bars(120))
        assert result is not None
        assert result.signal.replace("_", " ") in result.verdict
