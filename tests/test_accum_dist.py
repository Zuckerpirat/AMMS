"""Tests for amms.analysis.accum_dist."""

from __future__ import annotations

import pytest

from amms.analysis.accum_dist import ADIReport, analyze


class _Bar:
    def __init__(self, close: float, high: float = None, low: float = None, volume: float = 100_000):
        self.close = close
        self.high = high if high is not None else close + 1.0
        self.low = low if low is not None else close - 1.0
        self.volume = volume


def _accumulating(n: int = 40) -> list[_Bar]:
    """Close near high → MFM ≈ +0.75 → rising ADI."""
    bars = []
    for i in range(n):
        close = 100.0 + i * 0.2
        bars.append(_Bar(close, high=close + 0.5, low=close - 2.0, volume=100_000))
    return bars


def _distributing(n: int = 40) -> list[_Bar]:
    """Close near low → MFM ≈ -0.6 → falling ADI."""
    bars = []
    for i in range(n):
        close = 100.0 - i * 0.2
        # high = close+2, low = close-0.5 → MFM = (0.5 - 2.0)/2.5 = -0.6
        bars.append(_Bar(close, high=close + 2.0, low=close - 0.5, volume=100_000))
    return bars


def _bullish_divergence(n: int = 40) -> list[_Bar]:
    """Price falling but closes near high (accumulation under falling price)."""
    bars = []
    for i in range(n):
        close = 100.0 - i * 0.3
        # close near high: high=close+0.3, low=close-3.0 → MFM positive
        bars.append(_Bar(max(close, 1.0), high=max(close + 0.3, 2.0), low=max(close - 3.0, 0.1), volume=100_000))
    return bars


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(15)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(30))
        assert result is not None
        assert isinstance(result, ADIReport)

    def test_returns_none_no_volume(self):
        class _NoVol:
            close = high = low = 100.0
        assert analyze([_NoVol()] * 30) is None


class TestADITrend:
    def test_adi_trend_valid(self):
        for bars in [_flat(40), _accumulating(40), _distributing(40)]:
            result = analyze(bars)
            if result:
                assert result.adi_trend in {"rising", "falling", "flat"}

    def test_accumulating_adi_rising(self):
        result = analyze(_accumulating(40))
        assert result is not None
        assert result.adi_trend == "rising"

    def test_distributing_adi_falling(self):
        result = analyze(_distributing(40))
        assert result is not None
        assert result.adi_trend == "falling"


class TestPriceTrend:
    def test_price_trend_valid(self):
        for bars in [_flat(40), _accumulating(40), _distributing(40)]:
            result = analyze(bars)
            if result:
                assert result.price_trend in {"rising", "falling", "flat"}

    def test_accumulating_price_rising(self):
        result = analyze(_accumulating(40))
        assert result is not None
        assert result.price_trend == "rising"

    def test_distributing_price_falling(self):
        result = analyze(_distributing(40))
        assert result is not None
        assert result.price_trend == "falling"


class TestDivergence:
    def test_divergence_valid(self):
        for bars in [_flat(40), _accumulating(40)]:
            result = analyze(bars)
            if result:
                assert result.divergence in {"bullish", "bearish", "none"}

    def test_bullish_divergence_detected(self):
        result = analyze(_bullish_divergence(40))
        assert result is not None
        assert result.divergence == "bullish"

    def test_no_divergence_for_aligned(self):
        result = analyze(_accumulating(40))
        assert result is not None
        assert result.divergence == "none"

    def test_divergence_strength_in_range(self):
        result = analyze(_flat(40))
        assert result is not None
        assert 0.0 <= result.divergence_strength <= 1.0


class TestEMA:
    def test_adi_ema_is_float(self):
        result = analyze(_flat(30))
        assert result is not None
        assert isinstance(result.adi_ema, float)

    def test_adi_above_ema_is_bool(self):
        result = analyze(_flat(30))
        assert result is not None
        assert isinstance(result.adi_above_ema, bool)


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.history_adi) > 0

    def test_history_length_bounded(self):
        result = analyze(_flat(60), history_bars=20)
        assert result is not None
        assert len(result.history_adi) <= 20


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 40

    def test_current_price_correct(self):
        result = analyze(_flat(40, price=200.0))
        assert result is not None
        assert abs(result.current_price - 200.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(40), symbol="GS")
        assert result is not None
        assert result.symbol == "GS"

    def test_avg_mfm_in_range(self):
        result = analyze(_flat(40))
        assert result is not None
        assert -1.0 <= result.avg_mfm <= 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_adi(self):
        result = analyze(_flat(30))
        assert result is not None
        text = result.verdict.lower()
        assert "a/d" in text or "accumulation" in text or "distribution" in text
