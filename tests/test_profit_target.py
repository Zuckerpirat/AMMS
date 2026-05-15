"""Tests for amms.analysis.profit_target."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.profit_target import TargetProgress, compute


def _bar(sym: str, close: float, high: float = None, low: float = None,
         i: int = 0) -> Bar:
    h = high if high is not None else close * 1.01
    l = low if low is not None else close * 0.99
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, 100_000)


def _bars(sym: str, n: int, price: float = 100.0) -> list[Bar]:
    return [_bar(sym, price, price * 1.01, price * 0.99, i) for i in range(n)]


class TestCompute:
    def test_returns_none_empty_bars(self):
        assert compute("AAPL", 100.0, []) is None

    def test_returns_none_zero_entry(self):
        assert compute("AAPL", 0.0, _bars("AAPL", 20)) is None

    def test_returns_none_insufficient_bars(self):
        assert compute("AAPL", 100.0, _bars("AAPL", 5)) is None

    def test_returns_result(self):
        result = compute("AAPL", 100.0, _bars("AAPL", 20))
        assert result is not None
        assert isinstance(result, TargetProgress)

    def test_symbol_preserved(self):
        result = compute("TSLA", 150.0, _bars("TSLA", 20))
        assert result is not None
        assert result.symbol == "TSLA"

    def test_target_ordering(self):
        """1R < 2R < 3R, all above entry."""
        result = compute("AAPL", 100.0, _bars("AAPL", 20))
        assert result is not None
        assert result.entry_price < result.target_1r < result.target_2r < result.target_3r

    def test_stop_below_entry(self):
        result = compute("AAPL", 100.0, _bars("AAPL", 20))
        assert result is not None
        assert result.stop_1atr < result.entry_price

    def test_target_2r_is_entry_plus_2_atr(self):
        result = compute("AAPL", 100.0, _bars("AAPL", 20))
        assert result is not None
        expected = result.entry_price + 2 * result.atr
        assert result.target_2r == pytest.approx(expected, abs=0.01)

    def test_pnl_pct_positive_above_entry(self):
        bars = [_bar("AAPL", 110.0, 111.1, 108.9, i) for i in range(20)]
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.pnl_pct > 0

    def test_pnl_pct_negative_below_entry(self):
        bars = [_bar("AAPL", 90.0, 90.9, 89.1, i) for i in range(20)]
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.pnl_pct < 0

    def test_exceeded_2r_true_when_above_target(self):
        """If price is at 3x entry, it must exceed 2R target."""
        bars = [_bar("AAPL", 150.0, 151.5, 148.5, i) for i in range(20)]
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        if result.target_2r <= 150.0:
            assert result.exceeded_2r is True

    def test_exceeded_2r_false_at_entry(self):
        bars = _bars("AAPL", 20, price=100.0)
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.exceeded_2r is False

    def test_r_multiple_positive_for_gain(self):
        bars = [_bar("AAPL", 103.0, 104.0, 102.0, i) for i in range(20)]
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.r_multiple > 0

    def test_r_multiple_negative_for_loss(self):
        bars = [_bar("AAPL", 97.0, 98.0, 96.0, i) for i in range(20)]
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.r_multiple < 0

    def test_pct_to_2r_100_when_at_target(self):
        """When price == target_2r exactly, pct_to_2r = 100."""
        bars = _bars("AAPL", 20, price=100.0)
        result0 = compute("AAPL", 100.0, bars)
        assert result0 is not None
        # Now set price to exactly target_2r
        target = result0.target_2r
        bars2 = [_bar("AAPL", target, target * 1.01, target * 0.99, i)
                 for i in range(20)]
        result2 = compute("AAPL", 100.0, bars2)
        assert result2 is not None
        assert result2.pct_to_2r == pytest.approx(100.0, abs=5.0)

    def test_bars_used_correct(self):
        bars = _bars("AAPL", 25)
        result = compute("AAPL", 100.0, bars)
        assert result is not None
        assert result.bars_used == 25
