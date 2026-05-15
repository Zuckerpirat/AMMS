"""Tests for amms.analysis.stop_optimizer."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.stop_optimizer import StopSuggestion, suggest_stops, _atr


def _bar(sym: str, close: float, high: float = None, low: float = None,
         i: int = 0) -> Bar:
    h = high if high is not None else close * 1.02
    l = low if low is not None else close * 0.98
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, 100_000)


def _bars(sym: str, n: int, price: float = 100.0) -> list[Bar]:
    return [_bar(sym, price, price * 1.02, price * 0.98, i) for i in range(n)]


class TestAtr:
    def test_returns_none_insufficient(self):
        bars = _bars("AAPL", 10)
        assert _atr(bars, period=14) is None

    def test_returns_positive(self):
        bars = _bars("AAPL", 20)
        result = _atr(bars, period=14)
        assert result is not None
        assert result > 0

    def test_wider_range_higher_atr(self):
        narrow = [_bar("A", 100.0, 101.0, 99.0, i) for i in range(20)]
        wide = [_bar("A", 100.0, 110.0, 90.0, i) for i in range(20)]
        assert _atr(wide, 14) > _atr(narrow, 14)


class TestSuggestStops:
    def test_returns_none_empty_bars(self):
        assert suggest_stops("AAPL", 100.0, []) is None

    def test_returns_none_zero_entry(self):
        bars = _bars("AAPL", 20)
        assert suggest_stops("AAPL", 0.0, bars) is None

    def test_returns_none_insufficient_bars(self):
        bars = _bars("AAPL", 5)
        assert suggest_stops("AAPL", 100.0, bars) is None

    def test_returns_result(self):
        bars = _bars("AAPL", 20)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert isinstance(result, StopSuggestion)

    def test_symbol_preserved(self):
        bars = _bars("TSLA", 20)
        result = suggest_stops("TSLA", 150.0, bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_stop_ordering(self):
        """Conservative stop > standard > wide (all below entry)."""
        bars = _bars("AAPL", 20, price=100.0)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.stop_conservative >= result.stop_standard >= result.stop_wide

    def test_stops_below_entry(self):
        bars = _bars("AAPL", 20, price=100.0)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.stop_conservative <= result.entry_price
        assert result.stop_standard <= result.entry_price
        assert result.stop_wide <= result.entry_price

    def test_target_above_entry(self):
        bars = _bars("AAPL", 20, price=100.0)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.target_2r > result.entry_price

    def test_target_2r_is_2x_risk(self):
        """target_2r = entry + 2 * (entry - stop_standard)."""
        bars = _bars("AAPL", 20, price=100.0)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        expected = result.entry_price + 2 * (result.entry_price - result.stop_standard)
        assert result.target_2r == pytest.approx(expected, abs=0.01)

    def test_risk_pct_positive(self):
        bars = _bars("AAPL", 20)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.risk_standard_pct > 0

    def test_atr_pct_positive(self):
        bars = _bars("AAPL", 20)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.atr_pct > 0

    def test_stop_violated_when_price_below_stop(self):
        """Price much below entry → stop violated."""
        # Entry at 100, price at 80 → standard stop at ~100 - 1.5*ATR
        # With 2% range bars, ATR ~ 4, stop_std ~ 94
        bars = [_bar("AAPL", 80.0, 82.0, 78.0, i) for i in range(20)]
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.stop_violated is True

    def test_stop_not_violated_when_price_above(self):
        """Price well above entry → stop not violated."""
        bars = _bars("AAPL", 20, price=120.0)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.stop_violated is False

    def test_bars_used_correct(self):
        bars = _bars("AAPL", 25)
        result = suggest_stops("AAPL", 100.0, bars)
        assert result is not None
        assert result.bars_used == 25

    def test_stops_not_negative(self):
        """Stops should never go negative."""
        bars = _bars("AAPL", 20, price=1.0)  # very cheap stock
        result = suggest_stops("AAPL", 1.0, bars)
        assert result is not None
        assert result.stop_conservative > 0
        assert result.stop_standard > 0
        assert result.stop_wide > 0
