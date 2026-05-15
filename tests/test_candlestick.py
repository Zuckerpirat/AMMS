"""Tests for candlestick pattern detector."""

from __future__ import annotations

from amms.data.bars import Bar
from amms.features.candlestick import CandlePattern, _single_candle, _two_candle, detect_patterns


def _bar(open_: float, high: float, low: float, close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", open_, high, low, close, 1000.0)


class TestSingleCandle:
    def test_doji_detected(self) -> None:
        bar = _bar(100.0, 103.0, 97.0, 100.05)
        result = _single_candle(bar)
        assert result is not None
        assert result.name == "Doji"
        assert result.direction == "neutral"

    def test_hammer_detected(self) -> None:
        # Small body at top, long lower shadow
        bar = _bar(100.0, 101.5, 95.0, 101.0)  # body 1.0, range 6.5, lower shadow 5.0
        result = _single_candle(bar)
        assert result is not None
        assert result.name == "Hammer"
        assert result.direction == "bullish"

    def test_shooting_star_detected(self) -> None:
        # Small body at bottom, long upper shadow
        bar = _bar(100.0, 106.0, 99.5, 101.0)  # body 1.0, range 6.5, upper shadow 5.0
        result = _single_candle(bar)
        assert result is not None
        assert result.name == "Shooting Star"
        assert result.direction == "bearish"

    def test_bullish_marubozu(self) -> None:
        bar = _bar(100.0, 105.0, 100.0, 105.0)
        result = _single_candle(bar)
        assert result is not None
        assert "Marubozu" in result.name
        assert result.direction == "bullish"

    def test_bearish_marubozu(self) -> None:
        bar = _bar(105.0, 105.0, 100.0, 100.0)
        result = _single_candle(bar)
        assert result is not None
        assert "Marubozu" in result.name
        assert result.direction == "bearish"

    def test_confidence_in_range(self) -> None:
        bar = _bar(100.0, 103.0, 97.0, 100.05)
        result = _single_candle(bar)
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_zero_range_returns_none(self) -> None:
        bar = _bar(100.0, 100.0, 100.0, 100.0)
        result = _single_candle(bar)
        assert result is None


class TestTwoCandle:
    def test_bullish_engulfing(self) -> None:
        prev = _bar(105.0, 106.0, 98.0, 99.0)   # bearish
        curr = _bar(98.0, 108.0, 97.0, 107.0)    # bullish, larger
        result = _two_candle(prev, curr)
        assert result is not None
        assert result.name == "Bullish Engulfing"
        assert result.direction == "bullish"

    def test_bearish_engulfing(self) -> None:
        prev = _bar(99.0, 107.0, 98.0, 105.0)   # bullish
        curr = _bar(106.0, 107.0, 97.0, 98.0)   # bearish, larger
        result = _two_candle(prev, curr)
        assert result is not None
        assert result.name == "Bearish Engulfing"
        assert result.direction == "bearish"


class TestDetectPatterns:
    def test_returns_list(self) -> None:
        bars = [_bar(100.0, 103.0, 97.0, 100.05, i) for i in range(5)]
        result = detect_patterns(bars)
        assert isinstance(result, list)

    def test_empty_bars_returns_empty(self) -> None:
        assert detect_patterns([]) == []

    def test_pattern_is_dataclass(self) -> None:
        bars = [_bar(100.0, 103.0, 97.0, 100.05, i) for i in range(5)]
        result = detect_patterns(bars)
        for p in result:
            assert isinstance(p, CandlePattern)
