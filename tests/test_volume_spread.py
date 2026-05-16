"""Tests for amms.analysis.volume_spread."""

from __future__ import annotations

import pytest

from amms.analysis.volume_spread import VSABar, VSAReport, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float, volume: float = 100_000):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _flat(n: int = 30, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, price + 1.0, price - 1.0, price)] * n


def _effort_up_bars(n: int = 30) -> list[_Bar]:
    """Wide up bars closing near high with high volume."""
    bars = _flat(n - 5)
    for _ in range(5):
        bars.append(_Bar(100.0, 106.0, 99.5, 105.5, volume=250_000))
    return bars


def _up_thrust_bars(n: int = 30) -> list[_Bar]:
    """Wide up bars closing near low with high volume — bearish."""
    bars = _flat(n - 5)
    for _ in range(5):
        bars.append(_Bar(100.0, 106.0, 99.5, 100.5, volume=250_000))
    return bars


def _no_supply_bars(n: int = 30) -> list[_Bar]:
    """Narrow down bars (close<open) closing near high with low volume — bullish No Supply."""
    bars = _flat(n - 5)
    for _ in range(5):
        # open=101.5 close=101.4 → down bar; spread=0.6 (narrow vs avg 2.0); cp=0.67 (near high)
        bars.append(_Bar(101.5, 101.6, 101.0, 101.4, volume=30_000))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(15)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(25))
        assert result is not None
        assert isinstance(result, VSAReport)

    def test_returns_none_no_volume(self):
        class _NoVol:
            open = high = low = close = 100.0
        assert analyze([_NoVol()] * 25) is None


class TestCurrentBar:
    def test_current_bar_is_vsa_bar(self):
        result = analyze(_flat(25))
        assert result is not None
        assert isinstance(result.current_bar, VSABar)

    def test_current_bar_bias_valid(self):
        result = analyze(_flat(25))
        assert result is not None
        assert result.current_bar.bias in {"bullish", "bearish", "neutral"}

    def test_current_bar_strength_valid(self):
        result = analyze(_flat(25))
        assert result is not None
        assert result.current_bar.strength in {"strong", "moderate", "weak"}

    def test_current_bar_close_position_in_range(self):
        result = analyze(_flat(25))
        assert result is not None
        assert 0.0 <= result.current_bar.close_position <= 1.0


class TestRecentBars:
    def test_recent_bars_not_empty(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.recent_bars) > 0

    def test_recent_bars_are_vsa_bars(self):
        result = analyze(_flat(30))
        assert result is not None
        for b in result.recent_bars:
            assert isinstance(b, VSABar)

    def test_recent_bars_relative_vol_positive(self):
        result = analyze(_flat(30))
        assert result is not None
        for b in result.recent_bars:
            assert b.relative_volume >= 0


class TestSignals:
    def test_effort_up_detected(self):
        result = analyze(_effort_up_bars(30))
        assert result is not None
        labels = [b.label for b in result.recent_bars]
        assert "Effort Up" in labels

    def test_effort_up_bullish(self):
        result = analyze(_effort_up_bars(30))
        assert result is not None
        effort_bars = [b for b in result.recent_bars if b.label == "Effort Up"]
        assert all(b.bias == "bullish" for b in effort_bars)

    def test_up_thrust_detected(self):
        result = analyze(_up_thrust_bars(30))
        assert result is not None
        labels = [b.label for b in result.recent_bars]
        assert "Up Thrust" in labels

    def test_up_thrust_bearish(self):
        result = analyze(_up_thrust_bars(30))
        assert result is not None
        thrust_bars = [b for b in result.recent_bars if b.label == "Up Thrust"]
        assert all(b.bias == "bearish" for b in thrust_bars)

    def test_no_supply_detected(self):
        result = analyze(_no_supply_bars(30))
        assert result is not None
        labels = [b.label for b in result.recent_bars]
        assert "No Supply" in labels


class TestBias:
    def test_dominant_bias_valid(self):
        for bars in [_flat(30), _effort_up_bars(30)]:
            result = analyze(bars)
            if result:
                assert result.dominant_bias in {"bullish", "bearish", "neutral"}

    def test_bias_score_in_range(self):
        result = analyze(_flat(30))
        assert result is not None
        assert -1.0 <= result.bias_score <= 1.0

    def test_effort_up_bullish_bias(self):
        result = analyze(_effort_up_bars(30))
        assert result is not None
        assert result.bullish_count >= result.bearish_count

    def test_up_thrust_bearish_bias(self):
        result = analyze(_up_thrust_bars(30))
        assert result is not None
        assert result.bearish_count >= result.bullish_count


class TestSupplyDemand:
    def test_supply_demand_are_bool(self):
        result = analyze(_flat(30))
        assert result is not None
        assert isinstance(result.supply_detected, bool)
        assert isinstance(result.demand_detected, bool)

    def test_demand_detected_for_effort_up(self):
        result = analyze(_effort_up_bars(30))
        assert result is not None
        assert result.demand_detected

    def test_supply_detected_for_up_thrust(self):
        result = analyze(_up_thrust_bars(30))
        assert result is not None
        assert result.supply_detected


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_current_price_correct(self):
        result = analyze(_flat(30, price=150.0))
        assert result is not None
        assert abs(result.current_price - 150.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(30), symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_avg_spread_positive(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.avg_spread > 0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_vsa(self):
        result = analyze(_flat(30))
        assert result is not None
        assert "vsa" in result.verdict.lower() or "volume" in result.verdict.lower()
