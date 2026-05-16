"""Tests for amms.analysis.mean_reversion_band."""

from __future__ import annotations

import pytest

from amms.analysis.mean_reversion_band import MRBand, MRBReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _stretched_up(n: int = 40, base: float = 100.0, final: float = 130.0) -> list[_Bar]:
    """Price at base for most bars, then big jump at end."""
    bars = [_Bar(base)] * (n - 5)
    step = (final - base) / 5
    for i in range(5):
        bars.append(_Bar(base + step * (i + 1)))
    return bars


def _stretched_down(n: int = 40, base: float = 100.0, final: float = 70.0) -> list[_Bar]:
    """Price at base for most bars, then big drop at end."""
    bars = [_Bar(base)] * (n - 5)
    step = (final - base) / 5
    for i in range(5):
        bars.append(_Bar(base + step * (i + 1)))
    return bars


def _oscillating(n: int = 40) -> list[_Bar]:
    """Bouncing around mean — clear mean reversion."""
    bars = []
    for i in range(n):
        price = 100.0 + 3.0 * (1 if i % 4 < 2 else -1)
        bars.append(_Bar(price))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat(15)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _flat(35)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, MRBReport)


class TestZScore:
    def test_z_near_zero_for_flat(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert abs(result.z_score) < 0.1

    def test_z_positive_for_stretched_up(self):
        bars = _stretched_up(40)
        result = analyze(bars)
        assert result is not None
        assert result.z_score > 0

    def test_z_negative_for_stretched_down(self):
        bars = _stretched_down(40)
        result = analyze(bars)
        assert result is not None
        assert result.z_score < 0


class TestBands:
    def test_five_bands_returned(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert len(result.bands) == 5

    def test_bands_are_mrband(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        for b in result.bands:
            assert isinstance(b, MRBand)

    def test_bands_sorted_by_sigma(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        sigmas = [b.sigma for b in result.bands]
        assert sigmas == sorted(sigmas)

    def test_upper_band_above_mean(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        upper = next(b for b in result.bands if b.sigma == 2.0)
        mean_band = next(b for b in result.bands if b.sigma == 0.0)
        assert upper.price >= mean_band.price


class TestSignal:
    def test_signal_valid(self):
        for bars in [_flat(40), _stretched_up(40), _stretched_down(40)]:
            result = analyze(bars)
            if result:
                assert result.signal in {
                    "oversold_extreme", "oversold", "neutral", "overbought", "overbought_extreme"
                }

    def test_flat_is_neutral(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert result.signal == "neutral"

    def test_stretched_up_is_overbought(self):
        bars = _stretched_up(40)
        result = analyze(bars)
        assert result is not None
        assert result.signal in {"overbought", "overbought_extreme"}

    def test_stretched_down_is_oversold(self):
        bars = _stretched_down(40)
        result = analyze(bars)
        assert result is not None
        assert result.signal in {"oversold", "oversold_extreme"}

    def test_signal_strength_in_range(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.signal_strength <= 1.0


class TestPercentile:
    def test_z_percentile_in_range(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.z_percentile <= 100.0


class TestMetadata:
    def test_current_price_correct(self):
        bars = _flat(40, price=150.0)
        result = analyze(bars)
        assert result is not None
        assert abs(result.current_price - 150.0) < 0.01

    def test_bars_used_correct(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 40

    def test_symbol_stored(self):
        bars = _flat(40)
        result = analyze(bars, symbol="ETH")
        assert result is not None
        assert result.symbol == "ETH"


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_reversion(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert "reversion" in result.verdict.lower() or "mean" in result.verdict.lower()
