"""Tests for amms.analysis.multi_timeframe."""

from __future__ import annotations

import pytest

from amms.analysis.multi_timeframe import MultiTimeframeReport, TimeframeTrend, analyze


class _Bar:
    def __init__(self, high, low, close, symbol="SYM", volume=1_000_000):
        self.symbol = symbol
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _bullish(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start + i * step + 0.3, start + i * step - 0.3, start + i * step) for i in range(n)]


def _bearish(n: int = 60, start: float = 130.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start - i * step + 0.3, start - i * step - 0.3, start - i * step) for i in range(n)]


def _flat(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 0.3, price - 0.3, price) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = _bullish(10)
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, MultiTimeframeReport)


class TestTiers:
    def test_has_tiers(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert len(result.tiers) > 0

    def test_tier_labels(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        labels = [t.label for t in result.tiers]
        assert "Short" in labels
        assert "Medium" in labels
        assert "Long" in labels

    def test_tier_direction_valid(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        for t in result.tiers:
            assert t.direction in ("up", "down", "flat")

    def test_tier_strength_in_range(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        for t in result.tiers:
            assert 0 <= t.strength <= 100

    def test_tier_price_vs_ema_valid(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        for t in result.tiers:
            assert t.price_vs_ema in ("above", "below", "at")


class TestAlignment:
    def test_bullish_all_aligned(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert result.dominant_direction == "up"

    def test_bearish_all_aligned(self):
        bars = _bearish(60)
        result = analyze(bars)
        assert result is not None
        assert result.dominant_direction == "down"

    def test_alignment_score_in_range(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert 0 <= result.alignment_score <= 100

    def test_aligned_flag_true_for_trend(self):
        bars = _bullish(80)
        result = analyze(bars)
        assert result is not None
        assert result.aligned is True

    def test_current_price_correct(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)


class TestSymbol:
    def test_symbol_stored(self):
        bars = _bullish(60)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_available(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_available == 60


class TestVerdict:
    def test_verdict_present(self):
        bars = _bullish(60)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 5
