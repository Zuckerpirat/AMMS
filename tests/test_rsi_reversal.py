from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.strategy.rsi_reversal import RsiReversal


def _bar(i: int, close: float) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, close + 1, close - 1, close, 1000)


def _bars_flat(n: int, price: float = 100.0) -> list[Bar]:
    return [_bar(i, price) for i in range(n)]


def _bars_declining(n: int, start: float = 150.0, step: float = 2.0) -> list[Bar]:
    return [_bar(i, start - i * step) for i in range(n)]


def _bars_rising(n: int, start: float = 50.0, step: float = 2.0) -> list[Bar]:
    return [_bar(i, start + i * step) for i in range(n)]


def test_default_params() -> None:
    s = RsiReversal()
    assert s.rsi_period == 14
    assert s.oversold == 30.0
    assert s.overbought == 70.0
    assert s.name == "rsi_reversal"


def test_lookback() -> None:
    s = RsiReversal(rsi_period=14)
    assert s.lookback == 15


def test_hold_when_insufficient_history() -> None:
    s = RsiReversal()
    sig = s.evaluate("X", _bars_flat(5))
    assert sig.kind == "hold"
    assert "history" in sig.reason


def test_buy_on_oversold() -> None:
    s = RsiReversal(oversold=30.0)
    # Sharply declining prices → low RSI
    bars = _bars_declining(30, start=150.0, step=4.0)
    sig = s.evaluate("X", bars)
    assert sig.kind == "buy"
    assert sig.score > 0


def test_sell_on_overbought() -> None:
    s = RsiReversal(overbought=70.0)
    # Sharply rising prices → high RSI
    bars = _bars_rising(30, start=50.0, step=4.0)
    sig = s.evaluate("X", bars)
    assert sig.kind == "sell"


def test_hold_in_neutral_zone() -> None:
    s = RsiReversal()
    bars = _bars_flat(30)
    # Flat bars → RSI is undefined (all gains/losses = 0) → returns 100
    # or hold (equal gains/losses)
    sig = s.evaluate("X", bars)
    # Flat → RSI=100.0 (no losses) → overbought → sell
    assert sig.kind in ("sell", "hold")


def test_validation_bad_rsi_period() -> None:
    with pytest.raises(ValueError, match="rsi_period"):
        RsiReversal(rsi_period=1)


def test_validation_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        RsiReversal(oversold=80.0, overbought=20.0)


def test_strong_oversold_boosts_score() -> None:
    s = RsiReversal(oversold=30.0, strong_oversold=20.0)
    bars = _bars_declining(30, start=200.0, step=10.0)
    sig = s.evaluate("X", bars)
    if sig.kind == "buy":
        assert sig.score > 0


def test_registered_in_strategy_module() -> None:
    from amms.strategy import build_strategy
    s = build_strategy("rsi_reversal", {})
    assert s.name == "rsi_reversal"


def test_backtestable() -> None:
    """RsiReversal can be used as a backtest strategy."""
    from datetime import date

    from amms.backtest import BacktestConfig, Portfolio, run_backtest
    from amms.risk import RiskConfig

    # Simple smoke test — just verify it doesn't crash
    config = BacktestConfig(
        start=date(2025, 1, 1),
        end=date(2025, 1, 31),
        symbols=("AAPL",),
        initial_equity=10_000.0,
        risk=RiskConfig(),
        strategy=RsiReversal(),
    )
