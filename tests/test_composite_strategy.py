from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.strategy.composite import CompositeStrategy


def _bar(
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
) -> Bar:
    return Bar(
        symbol="X",
        timeframe="1Day",
        ts="2025-01-01T00:00:00Z",
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=volume,
    )


def test_rejects_non_positive_windows() -> None:
    with pytest.raises(ValueError):
        CompositeStrategy(momentum_n=0)


def test_rejects_inverted_sell_threshold() -> None:
    with pytest.raises(ValueError):
        CompositeStrategy(momentum_min=0.05, momentum_sell=0.10)


def test_hold_when_history_too_short() -> None:
    strat = CompositeStrategy()
    bars = [_bar(close=10) for _ in range(5)]
    sig = strat.evaluate("AAPL", bars)
    assert sig.kind == "hold"
    assert "bars" in sig.reason.lower()


def _uptrend_with_dips(volume_last: float = 200.0) -> list[Bar]:
    """Generate 21 bars: trend up ~11.5% with a few -2% dips so RSI stays moderate."""
    closes: list[float] = []
    val = 100.0
    for i in range(21):
        val *= 0.98 if i in {7, 13, 18} else 1.01
        closes.append(val)
    bars = [_bar(close=c, volume=100) for c in closes[:-1]]
    bars.append(_bar(close=closes[-1], volume=volume_last))
    return bars


def test_buy_when_all_filters_pass() -> None:
    bars = _uptrend_with_dips()
    sig = CompositeStrategy().evaluate("AAPL", bars)
    assert sig.kind == "buy", f"unexpected {sig}"
    assert sig.score > 0
    assert "composite ok" in sig.reason


def test_hold_when_momentum_below_threshold() -> None:
    bars = [_bar(close=100, volume=100) for _ in range(21)]
    strat = CompositeStrategy()
    sig = strat.evaluate("AAPL", bars)
    assert sig.kind == "hold"
    assert "momentum" in sig.reason


def test_sell_when_momentum_reverses() -> None:
    # 21 flat then a -10% drop on the last bar → 1-day momentum = -10%.
    closes = [100.0] * 20 + [90.0]
    bars = [_bar(close=c, volume=100) for c in closes]
    strat = CompositeStrategy(momentum_n=20, momentum_sell=-0.05)
    sig = strat.evaluate("AAPL", bars)
    assert sig.kind == "sell"
    assert "momentum" in sig.reason.lower()


def test_signal_carries_score() -> None:
    sig = CompositeStrategy().evaluate("AAPL", _uptrend_with_dips())
    assert sig.score > 0


def test_hold_when_rvol_below_threshold() -> None:
    # Same uptrend but the last bar's volume matches the average → rvol ≈ 1.0 < 1.2.
    bars = _uptrend_with_dips(volume_last=100.0)
    sig = CompositeStrategy().evaluate("AAPL", bars)
    assert sig.kind == "hold"
    assert "rvol" in sig.reason


def test_build_strategy_constructs_composite() -> None:
    from amms.strategy import build_strategy

    strat = build_strategy("composite", {"momentum_n": 10, "rsi_n": 14})
    assert isinstance(strat, CompositeStrategy)
    assert strat.momentum_n == 10


def test_sentiment_overlay_boosts_score() -> None:
    from amms.strategy.composite import set_sentiment_overlay

    bars = _uptrend_with_dips()
    base = CompositeStrategy().evaluate("AAPL", bars)
    assert base.kind == "buy"
    set_sentiment_overlay({"AAPL": 1.0})
    try:
        weighted = CompositeStrategy(sentiment_weight=0.5).evaluate("AAPL", bars)
        assert weighted.kind == "buy"
        assert weighted.score > base.score
    finally:
        set_sentiment_overlay({})


def test_sentiment_min_filter_blocks_buy() -> None:
    from amms.strategy.composite import set_sentiment_overlay

    bars = _uptrend_with_dips()
    set_sentiment_overlay({"AAPL": -0.9})
    try:
        sig = CompositeStrategy(sentiment_min=-0.5).evaluate("AAPL", bars)
        assert sig.kind == "hold"
        assert "sentiment" in sig.reason
    finally:
        set_sentiment_overlay({})
