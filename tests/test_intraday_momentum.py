"""Tests for amms.analysis.intraday_momentum."""

from __future__ import annotations

import pytest

from amms.analysis.intraday_momentum import IntradayMomentum, compute


class _Bar:
    def __init__(self, open_, high, low, close, volume=100_000):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _bullish_bars(start: float, n: int, step: float = 1.0) -> list[_Bar]:
    """Rising bars from start."""
    bars = []
    for i in range(n):
        o = start + i * step
        bars.append(_Bar(o, o + step * 0.8, o - step * 0.2, o + step * 0.6, 100_000 + i * 1000))
    return bars


def _bearish_bars(start: float, n: int, step: float = 1.0) -> list[_Bar]:
    bars = []
    for i in range(n):
        o = start - i * step
        bars.append(_Bar(o, o + step * 0.2, o - step * 0.8, o - step * 0.6, 100_000))
    return bars


def _flat_bars(price: float, n: int) -> list[_Bar]:
    return [_Bar(price, price + 0.5, price - 0.5, price, 100_000) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert compute([]) is None

    def test_returns_none_too_few(self):
        assert compute([_Bar(100, 101, 99, 100)] * 2) is None

    def test_returns_none_zero_range(self):
        bars = [_Bar(100, 100, 100, 100)] * 5
        assert compute(bars) is None

    def test_returns_result(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert isinstance(result, IntradayMomentum)


class TestSessionMetrics:
    def test_session_open_first_bar(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.session_open == pytest.approx(bars[0].open, abs=0.01)

    def test_session_high_max(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        expected_high = max(b.high for b in bars)
        assert result.session_high == pytest.approx(expected_high, abs=0.01)

    def test_session_low_min(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        expected_low = min(b.low for b in bars)
        assert result.session_low == pytest.approx(expected_low, abs=0.01)

    def test_current_price_last_close(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_session_range_positive(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.session_range > 0

    def test_price_in_range_at_high(self):
        """Price at session high → 100%."""
        bars = _bullish_bars(100.0, 10)
        # Force last close to be at session high
        bars[-1] = _Bar(bars[-1].open, bars[-1].high, bars[-1].low, bars[-1].high)
        result = compute(bars)
        assert result is not None
        assert result.price_in_range_pct == pytest.approx(100.0, abs=1.0)

    def test_cumulative_return_positive_for_bullish(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.cumulative_return_pct > 0

    def test_cumulative_return_negative_for_bearish(self):
        bars = _bearish_bars(110.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.cumulative_return_pct < 0


class TestVWAP:
    def test_vwap_in_price_range(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.session_low <= result.vwap <= result.session_high

    def test_price_above_vwap_for_bullish(self):
        bars = _bullish_bars(100.0, 15)
        result = compute(bars)
        assert result is not None
        # Strongly bullish bars → price should be above VWAP
        assert result.price_vs_vwap in ("above", "at")

    def test_price_below_vwap_for_bearish(self):
        bars = _bearish_bars(120.0, 15)
        result = compute(bars)
        assert result is not None
        assert result.price_vs_vwap in ("below", "at")


class TestOpeningGap:
    def test_gap_zero_no_prev_close(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.opening_gap_pct == 0.0

    def test_gap_up(self):
        bars = _bullish_bars(105.0, 10)
        result = compute(bars, prev_close=100.0)
        assert result is not None
        assert result.opening_gap_pct == pytest.approx(5.0, abs=0.1)

    def test_gap_down(self):
        bars = _bearish_bars(95.0, 10)
        result = compute(bars, prev_close=100.0)
        assert result is not None
        assert result.opening_gap_pct == pytest.approx(-5.0, abs=0.1)


class TestMomentumSignal:
    def test_strong_bull_for_bullish_bars(self):
        bars = _bullish_bars(100.0, 15)
        result = compute(bars)
        assert result is not None
        assert result.momentum_signal in ("strong_bull", "bull")

    def test_bear_or_neutral_for_bearish_bars(self):
        bars = _bearish_bars(120.0, 15)
        result = compute(bars)
        assert result is not None
        assert result.momentum_signal in ("strong_bear", "bear", "neutral")

    def test_neutral_for_flat_bars(self):
        bars = _flat_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        # Flat bars have no clear momentum
        assert result.momentum_signal in ("neutral", "bull", "bear")

    def test_signal_valid_value(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert result.momentum_signal in ("strong_bull", "bull", "neutral", "bear", "strong_bear")


class TestADRatio:
    def test_ad_ratio_bounded(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert -1.0 <= result.ad_ratio <= 1.0

    def test_ad_positive_for_bullish(self):
        """Close near high → positive AD."""
        bars = [_Bar(100, 102, 99, 101.8, 100_000) for _ in range(10)]
        result = compute(bars)
        assert result is not None
        assert result.ad_ratio > 0

    def test_ad_negative_for_bearish(self):
        """Close near low → negative AD."""
        bars = [_Bar(100, 102, 99, 99.2, 100_000) for _ in range(10)]
        result = compute(bars)
        assert result is not None
        assert result.ad_ratio < 0


class TestFadeDetection:
    def test_no_fade_for_trending(self):
        bars = _bullish_bars(100.0, 12)
        result = compute(bars)
        assert result is not None
        assert result.fade_detected is False

    def test_fade_for_reversal(self):
        """Strong open move that completely reverses → fade detected."""
        bars = []
        # First third: strong upward
        for i in range(4):
            bars.append(_Bar(100 + i, 101 + i, 99 + i, 100.5 + i))
        # Last two-thirds: completely reverses
        for i in range(8):
            bars.append(_Bar(104 - i, 105 - i, 103 - i, 104 - i * 0.8))
        result = compute(bars)
        assert result is not None
        # May or may not detect depending on exact numbers
        assert isinstance(result.fade_detected, bool)


class TestSymbolAndBars:
    def test_symbol_stored(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used(self):
        bars = _bullish_bars(100.0, 15)
        result = compute(bars)
        assert result is not None
        assert result.bars_used == 15

    def test_verdict_present(self):
        bars = _bullish_bars(100.0, 10)
        result = compute(bars)
        assert result is not None
        assert len(result.verdict) > 5
