from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.strategy.sma_cross import SmaCross


def _bars(closes: list[float], symbol: str = "AAPL") -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timeframe="1Day",
            ts=f"2025-01-{i + 1:02d}T05:00:00Z",
            open=c,
            high=c,
            low=c,
            close=c,
            volume=100,
        )
        for i, c in enumerate(closes)
    ]


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
    sig = strat.evaluate("AAPL", _bars([1.0, 2.0, 3.0]))
    assert sig.kind == "hold"
    assert "need" in sig.reason


def test_buy_on_upward_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    sig = strat.evaluate("AAPL", _bars([1, 1, 1, 1, 1, 1, 10]))
    assert sig.kind == "buy"
    assert sig.price == 10.0
    assert "above" in sig.reason


def test_sell_on_downward_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    sig = strat.evaluate("AAPL", _bars([10, 10, 10, 10, 10, 10, 1]))
    assert sig.kind == "sell"
    assert "below" in sig.reason


def test_hold_when_no_crossover() -> None:
    strat = SmaCross(fast=3, slow=5)
    sig = strat.evaluate("AAPL", _bars([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert sig.kind == "hold"


def test_lookback_matches_slow_plus_one() -> None:
    assert SmaCross(fast=10, slow=30).lookback == 31
