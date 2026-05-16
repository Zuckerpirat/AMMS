"""Tests for amms.analysis.heikin_ashi."""

from __future__ import annotations

import pytest

from amms.analysis.heikin_ashi import HACandle, HeikinAshiReport, analyze


class _Bar:
    def __init__(self, close, high=None, low=None, open_=None):
        self.close = close
        self.high = high if high is not None else close + 0.5
        self.low = low if low is not None else close - 0.5
        self.open = open_ if open_ is not None else close - 0.1


def _bullish(n: int = 30, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start + i * step, high=start + i * step + 0.8, low=start + i * step - 0.2) for i in range(n)]


def _bearish(n: int = 30, start: float = 115.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start - i * step, high=start - i * step + 0.2, low=start - i * step - 0.8) for i in range(n)]


def _flat(n: int = 30, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + (0.3 if i % 2 == 0 else -0.3)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(4)]
        assert analyze(bars) is None

    def test_returns_result_with_enough_bars(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, HeikinAshiReport)


class TestHACandles:
    def test_ha_candles_returned(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.ha_candles) > 0

    def test_ha_candles_are_hacandle(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        for c in result.ha_candles:
            assert isinstance(c, HACandle)

    def test_ha_candles_max_ten(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.ha_candles) <= 10

    def test_ha_high_gte_open_and_close(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        for c in result.ha_candles:
            assert c.ha_high >= c.ha_open
            assert c.ha_high >= c.ha_close

    def test_ha_low_lte_open_and_close(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        for c in result.ha_candles:
            assert c.ha_low <= c.ha_open
            assert c.ha_low <= c.ha_close

    def test_bullish_flag_set_correctly(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        for c in result.ha_candles:
            assert c.bullish == (c.ha_close > c.ha_open)


class TestTrend:
    def test_uptrend_detected(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert result.trend == "up"

    def test_downtrend_detected(self):
        bars = _bearish(30)
        result = analyze(bars)
        assert result is not None
        assert result.trend == "down"

    def test_trend_valid_for_flat(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.trend in ("up", "down", "reversal", "consolidating")

    def test_consecutive_bull_positive_in_uptrend(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert result.consecutive_bull >= 3

    def test_consecutive_bear_positive_in_downtrend(self):
        bars = _bearish(30)
        result = analyze(bars)
        assert result is not None
        assert result.consecutive_bear >= 3

    def test_only_one_run_nonzero(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        # In a clear trend, one run is active
        assert not (result.consecutive_bull > 0 and result.consecutive_bear > 0)


class TestTrendStrength:
    def test_strength_in_range(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.trend_strength <= 100.0

    def test_uptrend_strength_positive(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert result.trend_strength > 0

    def test_strong_candles_nonnegative(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert result.strong_candles >= 0


class TestCurrentPrice:
    def test_current_ha_open_and_close_set(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        assert result.current_ha_open > 0
        assert result.current_ha_close > 0

    def test_uptrend_ha_close_gt_ha_open(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        assert result.current_ha_close > result.current_ha_open

    def test_downtrend_ha_close_lt_ha_open(self):
        bars = _bearish(30)
        result = analyze(bars)
        assert result is not None
        assert result.current_ha_close < result.current_ha_open


class TestMetadata:
    def test_symbol_stored(self):
        bars = _bullish(20)
        result = analyze(bars, symbol="BTC")
        assert result is not None
        assert result.symbol == "BTC"

    def test_bars_used_correct(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 20

    def test_custom_lookback(self):
        bars = _bullish(30)
        result = analyze(bars, lookback=10)
        assert result is not None
        assert len(result.ha_candles) <= 10


class TestVerdict:
    def test_verdict_present(self):
        bars = _bullish(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_trend(self):
        bars = _bullish(30)
        result = analyze(bars)
        assert result is not None
        verdict_lower = result.verdict.lower()
        assert any(w in verdict_lower for w in ("up", "trend", "bull", "strength"))
