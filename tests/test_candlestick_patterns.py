"""Tests for amms.analysis.candlestick_patterns."""

from __future__ import annotations

import pytest

from amms.analysis.candlestick_patterns import CandlePattern, CandleReport, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close


def _bull_bar(base: float = 100.0, body: float = 5.0) -> _Bar:
    return _Bar(base, base + body + 1.0, base - 0.5, base + body)


def _bear_bar(base: float = 100.0, body: float = 5.0) -> _Bar:
    return _Bar(base + body, base + body + 0.5, base - 1.0, base)


def _doji_bar(price: float = 100.0) -> _Bar:
    return _Bar(price, price + 2.0, price - 2.0, price + 0.1)


def _hammer_bar(price: float = 100.0) -> _Bar:
    """Long lower wick, small body at top."""
    return _Bar(price + 3.5, price + 4.0, price, price + 4.0)


def _shooting_star_bar(price: float = 100.0) -> _Bar:
    """Long upper wick, small bear body at bottom."""
    return _Bar(price + 1.0, price + 9.0, price - 0.2, price)


def _marubozu_bull(base: float = 100.0) -> _Bar:
    return _Bar(base, base + 10.0, base, base + 10.0)


def _marubozu_bear(base: float = 110.0) -> _Bar:
    return _Bar(base, base, base - 10.0, base - 10.0)


def _flat_bars(n: int = 20, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, price + 1.0, price - 1.0, price)] * n


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_one_bar(self):
        assert analyze([_bull_bar()]) is None

    def test_returns_none_two_bars(self):
        assert analyze([_bull_bar(), _bear_bar()]) is None

    def test_returns_result_three_bars(self):
        bars = [_bull_bar(), _bear_bar(), _doji_bar()]
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, CandleReport)

    def test_returns_none_no_ohlc(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar(), _BadBar(), _BadBar()]) is None


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 20

    def test_current_price_correct(self):
        bars = _flat_bars(10, price=50.0)
        result = analyze(bars)
        assert result is not None
        assert abs(result.current_price - 50.0) < 0.5

    def test_symbol_stored(self):
        bars = _flat_bars(10)
        result = analyze(bars, symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"


class TestPatternTypes:
    def test_patterns_are_candle_pattern(self):
        bars = _flat_bars(5) + [_hammer_bar(), _doji_bar(), _shooting_star_bar()]
        result = analyze(bars)
        assert result is not None
        for p in result.patterns:
            assert isinstance(p, CandlePattern)

    def test_pattern_bias_valid(self):
        bars = _flat_bars(5) + [_hammer_bar(), _doji_bar(), _shooting_star_bar()]
        result = analyze(bars)
        assert result is not None
        for p in result.patterns:
            assert p.bias in {"bullish", "bearish", "neutral"}

    def test_pattern_strength_valid(self):
        bars = _flat_bars(5) + [_hammer_bar(), _doji_bar(), _shooting_star_bar()]
        result = analyze(bars)
        assert result is not None
        for p in result.patterns:
            assert p.strength in {"strong", "moderate", "weak"}


class TestSingleBarPatterns:
    def test_doji_detected(self):
        bars = _flat_bars(5) + [_doji_bar()]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert any("Doji" in n for n in names)

    def test_hammer_detected(self):
        bars = _flat_bars(5) + [_hammer_bar()]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert any("Hammer" in n or "hammer" in n for n in names)

    def test_shooting_star_detected(self):
        bars = _flat_bars(5) + [_shooting_star_bar()]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert any("Shooting Star" in n or "Inverted" in n for n in names)

    def test_bull_marubozu_detected(self):
        bars = _flat_bars(5) + [_marubozu_bull()]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert any("Marubozu" in n for n in names)

    def test_bear_marubozu_detected(self):
        bars = _flat_bars(5) + [_marubozu_bear()]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert any("Marubozu" in n for n in names)


class TestTwoBarPatterns:
    def test_bullish_engulfing_detected(self):
        # prev: bear from 110 to 100; current: bull from 98 to 113 (engulfs)
        prev = _Bar(110.0, 111.0, 99.0, 100.0)
        curr = _Bar(98.0, 115.0, 97.5, 113.0)
        bars = _flat_bars(5) + [prev, curr]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert "Bullish Engulfing" in names

    def test_bearish_engulfing_detected(self):
        # prev: bull from 100 to 110; current: bear from 112 to 98 (engulfs)
        prev = _Bar(100.0, 111.0, 99.0, 110.0)
        curr = _Bar(112.0, 113.0, 97.0, 98.0)
        bars = _flat_bars(5) + [prev, curr]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert "Bearish Engulfing" in names

    def test_engulfing_bullish_bias(self):
        prev = _Bar(110.0, 111.0, 99.0, 100.0)
        curr = _Bar(98.0, 115.0, 97.5, 113.0)
        bars = _flat_bars(5) + [prev, curr]
        result = analyze(bars)
        assert result is not None
        engulf = next((p for p in result.patterns if p.name == "Bullish Engulfing"), None)
        assert engulf is not None
        assert engulf.bias == "bullish"

    def test_engulfing_bearish_bias(self):
        prev = _Bar(100.0, 111.0, 99.0, 110.0)
        curr = _Bar(112.0, 113.0, 97.0, 98.0)
        bars = _flat_bars(5) + [prev, curr]
        result = analyze(bars)
        assert result is not None
        engulf = next((p for p in result.patterns if p.name == "Bearish Engulfing"), None)
        assert engulf is not None
        assert engulf.bias == "bearish"


class TestThreeBarPatterns:
    def test_three_white_soldiers_detected(self):
        bars = _flat_bars(5) + [
            _Bar(100.0, 106.0, 99.5, 105.0),
            _Bar(105.5, 112.0, 105.0, 111.0),
            _Bar(111.5, 118.0, 111.0, 117.0),
        ]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert "Three White Soldiers" in names

    def test_three_black_crows_detected(self):
        bars = _flat_bars(5) + [
            _Bar(117.0, 117.5, 111.0, 112.0),
            _Bar(111.5, 112.0, 105.0, 106.0),
            _Bar(105.5, 106.0, 99.0, 100.0),
        ]
        result = analyze(bars)
        assert result is not None
        names = [p.name for p in result.patterns]
        assert "Three Black Crows" in names

    def test_three_white_soldiers_strong(self):
        bars = _flat_bars(5) + [
            _Bar(100.0, 106.0, 99.5, 105.0),
            _Bar(105.5, 112.0, 105.0, 111.0),
            _Bar(111.5, 118.0, 111.0, 117.0),
        ]
        result = analyze(bars)
        assert result is not None
        pat = next((p for p in result.patterns if p.name == "Three White Soldiers"), None)
        assert pat is not None
        assert pat.strength == "strong"


class TestBiasScore:
    def test_bias_score_in_range(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert -1.0 <= result.bias_score <= 1.0

    def test_dominant_bias_valid(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.dominant_bias in {"bullish", "bearish", "neutral"}

    def test_bull_pattern_positive_bias(self):
        bars = _flat_bars(10) + [
            _Bar(110.0, 111.0, 99.0, 100.0),
            _Bar(98.0, 115.0, 97.5, 113.0),
        ]
        result = analyze(bars)
        assert result is not None
        # Bullish engulfing → at least some positive bias
        assert result.bullish_count >= 1


class TestCounts:
    def test_counts_non_negative(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.bullish_count >= 0
        assert result.bearish_count >= 0

    def test_counts_sum_le_total_patterns(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.bullish_count + result.bearish_count <= len(result.patterns)


class TestRecentPatterns:
    def test_recent_patterns_subset(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.recent_patterns) <= len(result.patterns)

    def test_recent_patterns_high_bar_idx(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        for p in result.recent_patterns:
            assert p.bar_idx >= len(bars) - 5


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat_bars(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_pattern_or_candle(self):
        bars = _flat_bars(5) + [_marubozu_bull()]
        result = analyze(bars)
        assert result is not None
        text = result.verdict.lower()
        assert "pattern" in text or "candle" in text or "bias" in text
