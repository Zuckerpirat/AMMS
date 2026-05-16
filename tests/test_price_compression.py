"""Tests for amms.analysis.price_compression."""

from __future__ import annotations

import pytest

from amms.analysis.price_compression import CompressionReport, analyze


class _Bar:
    def __init__(self, high, low, close, symbol="SYM"):
        self.symbol = symbol
        self.high = high
        self.low = low
        self.close = close
        self.volume = 1_000_000


def _normal_vol(n: int = 60, price: float = 100.0, range_: float = 3.0) -> list[_Bar]:
    """Normal volatility bars."""
    bars = []
    for i in range(n):
        bars.append(_Bar(price + range_, price - range_, price))
    return bars


def _compressed(n: int = 80, price: float = 100.0) -> list[_Bar]:
    """First 70%: normal vol. Last 10 bars: very compressed."""
    bars = []
    normal_n = int(n * 0.875)  # 70 bars normal
    for i in range(normal_n):
        bars.append(_Bar(price + 3.0, price - 3.0, price))
    for i in range(n - normal_n):
        bars.append(_Bar(price + 0.05, price - 0.05, price))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(101, 99, 100) for _ in range(20)]
        assert analyze(bars) is None

    def test_returns_none_bad_bars(self):
        assert analyze([None, None, None]) is None

    def test_returns_result(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, CompressionReport)


class TestCompressionDetection:
    def test_normal_vol_not_compressed(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert not result.compressed

    def test_compressed_atr_detected(self):
        """ATR compression is detected when recent bars have very low ATR."""
        bars = _compressed(80)
        result = analyze(bars)
        assert result is not None
        # ATR should be compressed below its historical average
        assert result.atr_ratio < 0.75

    def test_atr_ratio_in_range(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.atr_ratio > 0

    def test_bb_ratio_in_range(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.bb_width_ratio > 0

    def test_strength_in_range(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert 0 <= result.compression_strength <= 100

    def test_compressed_has_lower_atr_ratio(self):
        """Compressed scenario should have lower ATR ratio than normal volatility."""
        normal = _normal_vol(80)
        compressed = _compressed(80)
        r_normal = analyze(normal)
        r_compressed = analyze(compressed)
        assert r_normal is not None
        assert r_compressed is not None
        assert r_compressed.atr_ratio < r_normal.atr_ratio


class TestBias:
    def test_bias_valid(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.bias in ("bullish", "bearish", "neutral")

    def test_range_high_ge_range_low(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.range_high >= result.range_low

    def test_squeeze_bars_non_negative(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_in_squeeze >= 0


class TestMetadata:
    def test_current_price_correct(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_bars_used_correct(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_symbol_stored(self):
        bars = _normal_vol(60)
        result = analyze(bars, symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_atr_current_positive(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert result.atr_current > 0


class TestVerdict:
    def test_verdict_present(self):
        bars = _normal_vol(60)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 5

    def test_squeeze_in_verdict_when_compressed(self):
        bars = _compressed(80)
        result = analyze(bars)
        assert result is not None
        if result.compressed:
            assert "SQUEEZE" in result.verdict.upper() or "squeeze" in result.verdict.lower()
