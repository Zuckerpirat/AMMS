"""Tests for amms.analysis.wyckoff_phase."""

from __future__ import annotations

import pytest

from amms.analysis.wyckoff_phase import WyckoffReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0, volume: float = 100_000):
        self.close = close
        self.high = close + spread
        self.low = close - spread
        self.volume = volume


def _sideways(n: int = 60, price: float = 100.0) -> list[_Bar]:
    """Tight sideways range."""
    bars = []
    for i in range(n):
        offset = 1.0 * (1 if i % 4 < 2 else -1)
        bars.append(_Bar(price + offset))
    return bars


def _uptrend(n: int = 60, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend(n: int = 60, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(max(price, 1.0)))
        price -= step
    return bars


def _spring_pattern(n: int = 60, base: float = 100.0) -> list[_Bar]:
    """Sideways range, then brief dip below range, then recovery."""
    bars = _sideways(n - 10, price=base)
    # Dip below range
    for _ in range(3):
        bars.append(_Bar(base - 8.0, spread=1.0))
    # Recovery
    for _ in range(7):
        bars.append(_Bar(base + 2.0, spread=1.0))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_sideways(15)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_sideways(35))
        assert result is not None
        assert isinstance(result, WyckoffReport)


class TestPhase:
    _VALID_PHASES = {
        "accumulation_a", "accumulation_b", "accumulation_c",
        "markup", "distribution_a", "distribution_b",
        "distribution_c", "markdown", "trend_up", "trend_down", "unknown"
    }

    def test_phase_valid(self):
        for bars in [_sideways(60), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.phase in self._VALID_PHASES or result.phase in {
                    "accumulation_a", "accumulation_b", "accumulation_c",
                    "distribution_a", "distribution_b", "distribution_c",
                    "trend_up", "trend_down", "unknown"
                }

    def test_uptrend_trend_up(self):
        result = analyze(_uptrend(80, step=2.0))
        assert result is not None
        assert result.phase == "trend_up"

    def test_downtrend_trend_down(self):
        result = analyze(_downtrend(80, step=2.0))
        assert result is not None
        assert result.phase == "trend_down"

    def test_phase_label_present(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert len(result.phase_label) > 0


class TestConfidence:
    def test_confidence_in_range(self):
        for bars in [_sideways(60), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.confidence <= 1.0

    def test_uptrend_reasonable_confidence(self):
        result = analyze(_uptrend(80, step=2.0))
        assert result is not None
        assert result.confidence > 0.3


class TestRangeMetrics:
    def test_range_pct_non_negative(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert result.range_pct >= 0

    def test_price_position_in_range(self):
        for bars in [_sideways(60), _uptrend(60)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.price_position <= 1.0

    def test_range_high_ge_low(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert result.range_high >= result.range_low

    def test_sideways_is_sideways(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert result.is_sideways


class TestTrend:
    def test_trend_direction_valid(self):
        for bars in [_sideways(60), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.trend_direction in {"up", "down", "flat"}

    def test_uptrend_detected(self):
        result = analyze(_uptrend(80, step=2.0))
        assert result is not None
        assert result.trend_direction == "up"


class TestSpringUTAD:
    def test_spring_bool(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert isinstance(result.spring_detected, bool)

    def test_utad_bool(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert isinstance(result.utad_detected, bool)

    def test_spring_detected_for_spring_pattern(self):
        result = analyze(_spring_pattern(60))
        assert result is not None
        assert result.spring_detected


class TestVolume:
    def test_vol_climax_bool(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert isinstance(result.vol_climax, bool)

    def test_vol_drying_bool(self):
        result = analyze(_sideways(60))
        assert result is not None
        assert isinstance(result.vol_drying, bool)


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _sideways(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_current_price_positive(self):
        result = analyze(_sideways(60, price=100.0))
        assert result is not None
        assert result.current_price > 0

    def test_symbol_stored(self):
        result = analyze(_sideways(60), symbol="XOM")
        assert result is not None
        assert result.symbol == "XOM"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_sideways(35))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_wyckoff(self):
        result = analyze(_sideways(35))
        assert result is not None
        text = result.verdict.lower()
        assert "wyckoff" in text or "phase" in text or "accumulation" in text or "distribution" in text
