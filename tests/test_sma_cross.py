from __future__ import annotations

import pytest

from amms.strategy.sma_cross import SmaCross


def test_rejects_fast_geq_slow() -> None:
    with pytest.raises(ValueError):
        SmaCross(fast=10, slow=10)
    with pytest.raises(ValueError):
        SmaCross(fast=30, slow=10)


def test_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        SmaCross(fast=0, slow=10)


def test_hold_when_insufficient_bars() -> None:
    strat = SmaCross(fast=3, slow=5)
    sig = strat.evaluate("AAPL", [1.0, 2.0, 3.0])
    assert sig.kind == "hold"
    assert "need" in sig.reason


def test_buy_on_upward_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    # Six flat bars then a spike — fast SMA jumps above slow SMA only at the last bar.
    closes = [1, 1, 1, 1, 1, 1, 10]
    sig = strat.evaluate("AAPL", closes)
    assert sig.kind == "buy"
    assert sig.price == 10.0
    assert "above" in sig.reason


def test_sell_on_downward_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    closes = [10, 10, 10, 10, 10, 10, 1]
    sig = strat.evaluate("AAPL", closes)
    assert sig.kind == "sell"
    assert "below" in sig.reason


def test_hold_when_no_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    closes = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # monotonic up, fast stays above slow
    sig = strat.evaluate("AAPL", closes)
    assert sig.kind == "hold"


def test_lookback_matches_slow_plus_one() -> None:
    assert SmaCross(fast=10, slow=30).lookback == 31
