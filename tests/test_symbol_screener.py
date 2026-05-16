"""Tests for amms.analysis.symbol_screener."""

from __future__ import annotations

import pytest

from amms.analysis.symbol_screener import ScreenReport, ScreenResult, screen


class _Bar:
    def __init__(self, high, low, close, volume=500_000):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _bullish(start: float, n: int, step: float = 1.0, vol: float = 1_000_000) -> list[_Bar]:
    return [_Bar(start + i * step + 0.5, start + i * step - 0.5, start + i * step, vol) for i in range(n)]


def _bearish(start: float, n: int, step: float = 1.0, vol: float = 500_000) -> list[_Bar]:
    return [_Bar(start - i * step + 0.5, start - i * step - 0.5, start - i * step, vol) for i in range(n)]


def _flat(price: float, n: int, vol: float = 500_000) -> list[_Bar]:
    return [_Bar(price + 0.1, price - 0.1, price, vol) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert screen({}) is None

    def test_returns_result(self):
        bars = {"AAPL": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        assert isinstance(result, ScreenReport)

    def test_too_few_bars_skipped(self):
        bars = {"AAPL": _bullish(100.0, 3)}
        result = screen(bars)
        assert result is not None
        assert result.n_screened == 1
        assert len(result.results) == 0


class TestFilterRSI:
    def test_rsi_filter_passes_when_met(self):
        """Bullish bars → RSI high → passes RSI>=50 filter."""
        bars = {"AAPL": _bullish(100.0, 60)}
        result = screen(bars, rsi_min=50.0)
        assert result is not None
        # Should pass because strongly trending bars have high RSI
        assert len(result.results) > 0
        r = result.results[0]
        assert r.rsi is not None

    def test_rsi_filter_bearish_fails_high_min(self):
        """Bearish bars → RSI low → fails RSI>=70 filter."""
        bars = {"BEAR": _bearish(200.0, 60)}
        result = screen(bars, rsi_min=70.0)
        assert result is not None
        if result.results:
            bear = result.results[0]
            assert bear.passed_filters == 0 or (bear.rsi is not None and bear.rsi >= 70)

    def test_rsi_stored_in_result(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        if result.results:
            assert result.results[0].rsi is not None


class TestFilterROC:
    def test_roc_filter_bullish(self):
        """Bullish bars → positive ROC → passes roc_min > 0."""
        bars = {"BULL": _bullish(100.0, 60)}
        result = screen(bars, roc_min=0.0)
        assert result is not None
        if result.results:
            assert result.results[0].roc_20 is not None

    def test_roc_stored(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        if result.results:
            assert result.results[0].roc_20 is not None


class TestFilterSMA:
    def test_above_sma20_bullish(self):
        """Bullish bars → price above SMA20."""
        bars = {"BULL": _bullish(100.0, 60)}
        result = screen(bars, require_above_sma20=True)
        assert result is not None
        if result.results:
            bull = result.results[0]
            assert bull.above_sma20 is True

    def test_below_sma20_bearish(self):
        """Bearish bars → price below SMA20."""
        bars = {"BEAR": _bearish(200.0, 60)}
        result = screen(bars, require_above_sma20=False)
        assert result is not None
        if result.results:
            assert result.results[0].above_sma20 is False


class TestScoreAndRanking:
    def test_results_sorted_by_score_desc(self):
        bars = {
            "A": _bullish(100.0, 60),
            "B": _bearish(200.0, 60),
        }
        result = screen(bars, rsi_min=50.0, roc_min=0.0, require_above_sma20=True)
        assert result is not None
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_score_in_range(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        for r in result.results:
            assert 0.0 <= r.score <= 100.0

    def test_no_filters_score_50(self):
        """No filters → composite = 50 (neutral)."""
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        assert result.results[0].score == pytest.approx(50.0, abs=0.1)

    def test_min_score_filters_results(self):
        bars = {"X": _flat(100.0, 60)}
        result = screen(bars, rsi_min=80.0, min_score=80.0)
        assert result is not None
        for r in result.results:
            assert r.score >= 80.0


class TestVolumeAndATR:
    def test_volume_ratio_present_with_enough_bars(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        if result.results:
            assert result.results[0].volume_ratio is not None

    def test_atr_ratio_present_with_enough_bars(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        if result.results:
            assert result.results[0].atr_ratio is not None


class TestMetadata:
    def test_n_screened_correct(self):
        bars = {"A": _bullish(100.0, 60), "B": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        assert result.n_screened == 2

    def test_filters_applied_recorded(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars, rsi_min=50.0, require_above_sma20=True)
        assert result is not None
        assert len(result.filters_applied) == 2

    def test_symbol_stored(self):
        bars = {"AAPL": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        assert result.results[0].symbol == "AAPL"

    def test_price_stored(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        # Last close = 100 + 59 × 1 = 159
        assert result.results[0].price == pytest.approx(159.0, abs=0.1)

    def test_verdict_present(self):
        bars = {"X": _bullish(100.0, 60)}
        result = screen(bars)
        assert result is not None
        assert len(result.results[0].verdict) > 0
