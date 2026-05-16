"""Tests for amms.analysis.anchored_vwap."""

from __future__ import annotations

import pytest

from amms.analysis.anchored_vwap import AVWAPReport, analyze


class _Bar:
    def __init__(self, close, high=None, low=None, volume=1000.0):
        self.close = close
        self.high = high if high is not None else close + 0.5
        self.low = low if low is not None else close - 0.5
        self.volume = volume


def _rising(n: int = 40, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start + i * step, high=start + i * step + 0.6, low=start + i * step - 0.1) for i in range(n)]


def _falling(n: int = 40, start: float = 120.0, step: float = 0.5) -> list[_Bar]:
    return [_Bar(start - i * step, high=start - i * step + 0.1, low=start - i * step - 0.6) for i in range(n)]


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, volume=500.0) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(4)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _rising(20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, AVWAPReport)

    def test_invalid_anchor_returns_none(self):
        bars = _rising(20)
        result = analyze(bars, anchor="invalid_mode")
        assert result is None


class TestAVWAPValue:
    def test_flat_avwap_equals_price(self):
        bars = _flat(30, price=100.0)
        result = analyze(bars, anchor=20)
        assert result is not None
        assert result.avwap == pytest.approx(100.0, abs=0.01)

    def test_avwap_positive(self):
        bars = _rising(30)
        result = analyze(bars)
        assert result is not None
        assert result.avwap > 0

    def test_avwap_in_price_range(self):
        bars = _rising(30)
        prices = [b.close for b in bars]
        result = analyze(bars)
        assert result is not None
        assert min(prices) <= result.avwap <= max(prices) + 1


class TestBands:
    def test_upper_bands_above_avwap(self):
        bars = _rising(30)
        result = analyze(bars)
        assert result is not None
        assert result.upper_1 >= result.avwap
        assert result.upper_2 >= result.upper_1

    def test_lower_bands_below_avwap(self):
        bars = _rising(30)
        result = analyze(bars)
        assert result is not None
        assert result.lower_1 <= result.avwap
        assert result.lower_2 <= result.lower_1

    def test_flat_bars_zero_sigma(self):
        bars = _flat(30, price=100.0)
        result = analyze(bars, anchor=20)
        assert result is not None
        # Flat price → sigma ≈ 0 → all bands near AVWAP
        assert result.upper_1 == pytest.approx(result.avwap, abs=0.01)
        assert result.lower_1 == pytest.approx(result.avwap, abs=0.01)


class TestPricePosition:
    def test_rising_price_above_avwap(self):
        bars = _rising(40)
        result = analyze(bars, anchor=30)
        assert result is not None
        # In a rising market with 30-bar anchor, recent price > AVWAP
        assert result.price_position in ("above", "at")

    def test_falling_price_below_avwap(self):
        bars = _falling(40)
        result = analyze(bars, anchor=30)
        assert result is not None
        assert result.price_position in ("below", "at")

    def test_price_position_valid(self):
        bars = _rising(20)
        result = analyze(bars)
        assert result is not None
        assert result.price_position in ("above", "below", "at")

    def test_pct_from_avwap_sign_matches_position(self):
        bars = _rising(40)
        result = analyze(bars, anchor=30)
        assert result is not None
        if result.price_position == "above":
            assert result.pct_from_avwap > 0
        elif result.price_position == "below":
            assert result.pct_from_avwap < 0


class TestAnchors:
    def test_auto_low_anchor(self):
        bars = _rising(30)
        result = analyze(bars, anchor="auto_low")
        assert result is not None
        assert "swing low" in result.anchor_label

    def test_auto_high_anchor(self):
        bars = _falling(30)
        result = analyze(bars, anchor="auto_high")
        assert result is not None
        assert "swing high" in result.anchor_label

    def test_integer_anchor(self):
        bars = _rising(30)
        result = analyze(bars, anchor=10)
        assert result is not None
        assert "10 bars back" in result.anchor_label
        assert result.bars_in_window == 10

    def test_integer_anchor_clipped(self):
        bars = _rising(10)
        result = analyze(bars, anchor=100)
        assert result is not None
        assert result.bars_in_window >= 1


class TestMetadata:
    def test_symbol_stored(self):
        bars = _rising(20)
        result = analyze(bars, symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_total_bars_correct(self):
        bars = _rising(25)
        result = analyze(bars)
        assert result is not None
        assert result.total_bars == 25

    def test_current_price_matches_last_bar(self):
        bars = _rising(20)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_bars_in_window_le_total(self):
        bars = _rising(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_in_window <= result.total_bars


class TestVerdict:
    def test_verdict_present(self):
        bars = _rising(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_contains_avwap_value(self):
        bars = _rising(20)
        result = analyze(bars)
        assert result is not None
        assert str(round(result.avwap, 1)) in result.verdict or "AVWAP" in result.verdict
